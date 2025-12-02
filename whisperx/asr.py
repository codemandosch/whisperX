import os
from typing import List, Optional, Union
from dataclasses import replace

import ctranslate2
import faster_whisper
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_ctranslate2_storage
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisperx.schema import SingleSegment, TranscriptionResult
from whisperx.vads import Vad, Silero, Pyannote
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens


def get_silence_in_seconds(vad_segments: List[dict]):
    prev = 0
    silence = 0
    for seg in vad_segments:
        silence += seg["start"] - prev
        prev = seg["end"]
    return silence


class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    @staticmethod
    def merge_punctuations(
        alignment: List[dict], prepended: str, appended: str
    ) -> None:
        # merge prepended punctuations
        i = len(alignment) - 2
        j = len(alignment) - 1
        while i >= 0:
            previous = alignment[i]
            following = alignment[j]
            if (
                previous["word"].startswith(" ")
                and previous["word"].strip() in prepended
            ):
                # prepend it to the following word
                following["word"] = previous["word"] + following["word"]
                following["tokens"] = previous["tokens"] + following["tokens"]
                previous["word"] = ""
                previous["tokens"] = []
            else:
                j = i
            i -= 1

        # merge appended punctuations
        i = 0
        j = 1
        while j < len(alignment):
            previous = alignment[i]
            following = alignment[j]
            if not previous["word"].endswith(" ") and following["word"] in appended:
                # append it to the previous word
                previous["word"] = previous["word"] + following["word"]
                previous["tokens"] = previous["tokens"] + following["tokens"]
                following["word"] = ""
                following["tokens"] = []
            else:
                i = j
            j += 1
        return alignment

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        segments: List[List[tuple]] = None,
        padding_batch: List[int] = None,
        encoder_output=None,
    ):
        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            return tokenizer.tokenizer.decode_batch(res)

        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
            hotwords=options.hotwords
        )

        encoder_output = self.encode(features)

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                return_scores=True,
                beam_size=options.beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
            )

        tokens_batch = []
        avg_probs = []
        for res in result:
            tokens = res.sequences_ids[0]
            tokens_batch.append(tokens)

            # Calculate average logprob
            seq_len = len(tokens)
            cum_logprob = res.scores[0] * (seq_len**options.length_penalty)
            avg_logprob = cum_logprob / (seq_len + 1)
            avg_prob = torch.exp(torch.tensor(avg_logprob)).item()
            avg_probs.append(avg_prob)

        if options.word_timestamps and padding_batch is not None:
            segment_sizes = []
            for tokens, padding in zip(tokens_batch, padding_batch):
                # get all the segment sizes for each batch
                content_frames = features.shape[-1]
                segment_size = min(self.feature_extractor.nb_max_frames, content_frames)
                segment_sizes.append(int(segment_size - padding))

            # align the batches to get word timestamps
            alignments = self.find_alignment(
                tokenizer=tokenizer,
                tokens_batch=tokens_batch,
                encoder_output=encoder_output,
                num_frames=segment_sizes,
                vad_segments=segments,
            )
        else:
            alignments = None

        text = decode_batch(tokens_batch)
        return {"text": text, "avg_probs": avg_probs, "alignments": alignments}

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

    def find_alignment(
        self,
        tokenizer: Tokenizer,
        tokens_batch: List[List[int]],
        encoder_output: ctranslate2.StorageView,
        num_frames: List[int],
        median_filter_width: int = 7,
        vad_segments: List[List[tuple]] = None,
    ) -> List[dict]:
        results = self.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            tokens_batch,
            num_frames,
            median_filter_width=median_filter_width,
        )
        all_alignments = []
        for ind, result in enumerate(results):
            text_token_probs = result.text_token_probs

            alignments = result.alignments
            text_indices = np.array([pair[0] for pair in alignments])
            time_indices = np.array([pair[1] for pair in alignments])

            words, word_tokens = tokenizer.split_to_word_tokens(
                tokens_batch[ind] + [tokenizer.eot]
            )

            if len(word_tokens) <= 1:
                # TODO: fix this
                # return on eot only
                # >>> np.pad([], (1, 0))
                # array([0.])
                # This results in crashes when we lookup jump_times with float, like
                # IndexError: arrays used as indices must be of integer (or boolean) type
                return []
            word_boundaries = np.pad(
                np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0)
            )

            if len(word_boundaries) <= 1:
                return []

            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(
                bool
            )
            jump_times = time_indices[jumps] / self.tokens_per_second
            start_times = jump_times[word_boundaries[:-1]]
            end_times = jump_times[word_boundaries[1:]]

            word_probabilities = [
                np.mean(text_token_probs[i:j])
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
            ]

            all_alignments.append(
                [
                    dict(
                        word=word,
                        tokens=tokens,
                        start=start,
                        end=end,
                        probability=probability,
                    )
                    for word, tokens, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    )
                ]
            )

        new_alignments = []
        for alignment in all_alignments:
            word_durations = np.array(
                [word["end"] - word["start"] for word in alignment]
            )
            word_durations = word_durations[word_durations.nonzero()]
            median_duration = (
                np.median(word_durations) if len(word_durations) > 0 else 0.0
            )
            max_duration = median_duration * 2

            # hack: truncate long words at sentence boundaries.
            if len(word_durations) > 0:
                sentence_end_marks = ".。!！?？"
                # ensure words at sentence boundaries
                # are not longer than twice the median word duration.
                for i in range(1, len(alignment)):
                    if alignment[i]["end"] - alignment[i]["start"] > max_duration:
                        if alignment[i]["word"] in sentence_end_marks:
                            alignment[i]["end"] = alignment[i]["start"] + max_duration
                        elif alignment[i - 1]["word"] in sentence_end_marks:
                            alignment[i]["start"] = alignment[i]["end"] - max_duration
                        else:
                            alignment[i]["end"] = alignment[i]["start"] + max_duration

            prepend_punctuations: str = ("\"'"¿([{-",)
            append_punctuations: str = ("\"'.。,，!！?？:：")]}、",)
            alignment = self.merge_punctuations(
                alignment, prepend_punctuations, append_punctuations
            )

            new_alignments.append(alignment)
        return new_alignments


class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
        self,
        model: WhisperModel,
        vad,
        vad_params: dict,
        options: TranscriptionOptions,
        tokenizer: Optional[Tokenizer] = None,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        segments = audio.get("segments")
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features, num_padded_frames = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features, 'segments': segments, 'padding': num_padded_frames}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(
            model_inputs['inputs'],
            self.tokenizer,
            self.options,
            model_inputs.get('segments'),
            padding_batch=model_inputs.get('padding'),
        )
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params: dict,
        forward_params: dict,
        postprocess_params: dict,
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {
                'inputs': torch.stack([x['inputs'] for x in items]),
                'segments': [x.get('segments') for x in items],
                'padding': [x.get('padding') for x in items],
            }
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def calculate_audio_silence(self, audio: Union[str, np.ndarray], chunk_size=30):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_len = len(audio)

        # Pre-process audio and merge chunks as defined by the respective VAD child class
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks_fn = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks_fn = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks_fn(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        silence = get_silence_in_seconds(vad_segments)
        silence_in_seconds = silence * SAMPLE_RATE / audio_len

        return {"silence": silence, "silence_percentage": silence_in_seconds}

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        verbose=False,
        word_timestamps=False,
        initial_prompt=None,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_len = len(audio)

        # add word_timestamp and initial_prompt to options
        self.options = replace(self.options, word_timestamps=word_timestamps)
        self.options = replace(self.options, initial_prompt=initial_prompt)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2], 'segments': seg.get('segments')}

        # Pre-process audio and merge chunks as defined by the respective VAD child class 
        # In case vad_model is manually assigned (see 'load_model') follow the functionality of pyannote toolkit
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks =  self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        silence = get_silence_in_seconds(vad_segments)
        silence_in_seconds = silence * SAMPLE_RATE / audio_len
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            logger.info("Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = replace(self.options, suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            avg_prob = out['avg_probs']
            alignment = out.get('alignments')

            starting_time = round(vad_segments[idx]['start'], 3)

            if batch_size in [0, 1, None]:
                text = text[0]
                avg_prob = avg_prob[0]
                if alignment is not None:
                    alignment = alignment[0]

            if alignment is not None:
                for i in range(len(alignment)):
                    # add starting time to each word as the words start at 0 at each segment
                    alignment[i]["start"] += starting_time
                    alignment[i]["end"] += starting_time

            if verbose:
                print(f"Transcript: [{starting_time} --> {round(vad_segments[idx]['end'], 3)}] {text}")
            segments.append(
                {
                    "text": text,
                    "avg_prob": avg_prob,
                    "start": starting_time,
                    "end": round(vad_segments[idx]['end'], 3),
                    "words": alignment,
                }
            )

        # revert the tokenizer if multilingual inference is enabled
        # if self.preset_language is None:
        #     self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = replace(self.options, suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": language, "silence": silence_in_seconds}

    def detect_language(self, audio: np.ndarray) -> str:
        if audio.shape[0] < N_SAMPLES:
            logger.warning("Audio is shorter than 30s, language detection may be inaccurate")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        logger.info(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio")
        return language


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad]= None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    model: Optional[WhisperModel] = None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
) -> FasterWhisperPipeline:
    """Load a Whisper model for inference.
    Args:
        whisper_arch - The name of the Whisper model to load.
        device - The device to load the model on.
        compute_type - The compute type to use for the model.
        vad_model - The vad model to manually assign.
        vad_method - The vad method to use. vad_model has a higher priority if it is not None.
        options - A dictionary of options to use for the model.
        language - The language of the model. (use English for now)
        model - The WhisperModel instance to use.
        download_root - The root directory to download the model to.
        local_files_only - If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        threads - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    """

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         local_files_only=local_files_only,
                         cpu_threads=threads)
    if language is not None:
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        logger.info("No language specified, language will be detected for each audio file (increases inference time)")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "multilingual": model.model.is_multilingual,
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "chunk_size": 30, # needed by silero since binarization happens before merge_chunks
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    # Note: manually assigned vad_model has higher priority than vad_method!
    if vad_model is not None:
        print("Use manually assigned vad_model. vad_method is ignored.")
        vad_model = vad_model
    else:
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            if device == 'cuda':
                device_vad = f'cuda:{device_index}'
            else:
                device_vad = device
            vad_model = Pyannote(torch.device(device_vad), use_auth_token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )
