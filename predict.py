import os
import shutil
import subprocess
import tempfile
import warnings

import torch
import torchaudio
from cog import BasePredictor, Input, Path

from tortoise.api import MODELS_DIR, TextToSpeech

from tortoise.inference import (
        get_seed,
        split_text,
        voice_loader,
        save_gen_with_voicefix
    )

warnings.filterwarnings(
    "ignore"
)  # pysndfile does not support mp3, so we silence its warnings

VOICE_OPTIONS = [
    "angie",
    "cond_latent_example",
    "deniro",
    "freeman",
    "halle",
    "lj",
    "myself",
    "pat2",
    "snakes",
    "tom",
    "train_daws",
    "train_dreams",
    "train_grace",
    "train_lescault",
    "weaver",
    "applejack",
    "daniel",
    "emma",
    "geralt",
    "jlaw",
    "mol",
    "pat",
    "rainbow",
    "tim_reynolds",
    "train_atkins",
    "train_dotrice",
    "train_empire",
    "train_kennard",
    "train_mouse",
    "william",
    "random",  # special option for random voice
    "custom_voice",  # special option for custom voice
    "disabled",  # special option for disabled voice
]

MODULE_DIRECTORY = os.path.dirname(__file__)
CUSTOM_VOICE_DIRECTORY = Path(MODULE_DIRECTORY, "tortoise", "voices", "custom_voice")


def create_custom_voice_from_mp3(input_path: str, segment_time: int = 9) -> None:
    CUSTOM_VOICE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    normalized_audio_path = tempfile.mktemp(suffix=".wav")
    subprocess.check_output(
        [
            "ffmpeg-normalize",
            input_path,
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-o",
            normalized_audio_path,
        ]
    )
    assert os.path.exists(normalized_audio_path), "ffmpeg-normalize failed"
    subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "warning",  # log only errors
            "-ss",
            "00:00:00",
            "-t",
            "300",  # limit to 300 seconds (~30 segments)
            "-i",
            normalized_audio_path,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-f",
            "segment",
            "-segment_time",
            str(segment_time),
            os.path.join(CUSTOM_VOICE_DIRECTORY, "%d.wav"),
        ]
    )


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.text_to_speech = TextToSpeech(
            models_dir=MODELS_DIR,
            enable_redaction=True,
            device="cuda",
            autoregressive_batch_size=16,
            high_vram=True,
            kv_cache=True,
            ar_checkpoint=None, # None means use the default checkpoint
            clvp_checkpoint=None, # None means use the default checkpoint
            diff_checkpoint=None, # None means use the default checkpoint
        )

    def predict(
        self,
        text: str = Input(
            description="Text to speak.",
            default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.",
        ),
        voice_a: str = Input(
            description="Selects the voice to use for generation. Use `random` to select a random voice. Use `custom_voice` to use a custom voice.",
            default="random",
            choices=VOICE_OPTIONS,
        ),
        custom_voice: Path = Input(
            description="(Optional) Create a custom voice based on an mp3 file of a speaker. Audio should be at least 15 seconds, only contain one speaker, and be in mp3 format. Overrides the `voice_a` input.",
            default=None,
        ),
        voice_b: str = Input(
            description="(Optional) Create new voice from averaging the latents for `voice_a`, `voice_b` and `voice_c`. Use `disabled` to disable voice mixing.",
            default="disabled",
            choices=VOICE_OPTIONS,
        ),
        voice_c: str = Input(
            description="(Optional) Create new voice from averaging the latents for `voice_a`, `voice_b` and `voice_c`. Use `disabled` to disable voice mixing.",
            default="disabled",
            choices=VOICE_OPTIONS,
        ),
        preset: str = Input(
            description="Which voice preset to use. See the documentation for more information.",
            default="ultra_fast",
            choices=["ultra_fast", "fast", "standard", "high_quality"],
        ),
        seed: int = Input(
            description="Random seed which can be used to reproduce results.",
            default=0,
        ),
        cvvp_amount: float = Input(
            description="How much the CVVP model should influence the output. Increasing this can in some cases reduce the likelyhood of multiple speakers. Defaults to 0 (disabled)",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        temperature: float = Input(
            description="Temperature for sampling. Lower values are more deterministic, higher values are more random.",
            default=0.2,
        ),
        length_penalty: float = Input(
            description="A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs.",
            default=1.0,
        ),
        repetition_penalty: float = Input(
            description="A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence of long silences or 'uhhhhhhs', etc.",
            default=2.0,
        ),
        top_p: float = Input(
            description="P value used in nucleus sampling. 0 to 1. Lower values mean the decoder produces more 'likely' (aka boring) outputs. Higher values mean the decoder produces more 'unlikely' (aka interesting) outputs.",
            default=0.8,
            ge=0.0,
            le=1.0,
        ),
        diffusion_temperature: float = Input(
            description="Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0 are the 'mean' prediction of the diffusion network and will sound bland and smeared.",
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        original_tortoise: bool = Input(
            description="ensure results are identical to original tortoise-tts repo but will be slower",
            default=False,
        ),
        use_voicefixer: bool = Input(
            description="Use voicefixer to improve the quality of the generated audio",
            default=True,
        ),
                    
    ) -> Path:
        
        output_dir = Path(tempfile.mkdtemp())

        if custom_voice is not None:
            assert (
                custom_voice.suffix == ".mp3"
            ), f"File {custom_voice} is not an mp3 file"
            print(f"Creating voice from {custom_voice}")
            # remove the custom voice dir if it exists
            shutil.rmtree(str(CUSTOM_VOICE_DIRECTORY), ignore_errors=True)
            create_custom_voice_from_mp3(str(custom_voice))
            all_voices = ["custom_voice"]
        else:
            all_voices = [voice_a]
        if voice_b != "disabled":
            all_voices.append(voice_b)
        if voice_c != "disabled":
            all_voices.append(voice_c)
        print(f"Generating text using voices: {all_voices}")
        
        # get the voices
        selected_voices = [all_voices]
        voice_generator = voice_loader(selected_voices, extra_voice_dirs=[])
        
        # handle text
        texts = split_text(text, text_split=None)
        seed = get_seed(seed)
        
        total_clips = len(texts) * len(selected_voices)
        
        # generation setting
        verbose = True
        gen_settings = {
            "use_deterministic_seed": seed,
            "verbose": verbose,
            "k": 1,
            "preset": preset,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "diffusion_temperature": diffusion_temperature,
            "cvvp_amount": cvvp_amount,
            "original_tortoise": original_tortoise,
        }
        num_candidates = 1
        
        final_file_output = None
        for voice_idx, (voice, voice_samples, conditioning_latents) in enumerate(
            voice_generator
        ):
            audio_parts = []
            for text_idx, text in enumerate(texts):
                clip_name = f'{"-".join(voice)}_{text_idx:02d}'
                first_clip = output_dir.joinpath(f"{clip_name}_00.wav") 
                    
                if verbose:
                    print(
                        f"Rendering {clip_name} ({(voice_idx * len(texts) + text_idx + 1)} of {total_clips})..."
                    )
                    print("  " + text)
                    
                gen = self.text_to_speech.tts_with_preset(
                    text,
                    voice_samples=voice_samples,
                    conditioning_latents=conditioning_latents,
                    **gen_settings,
                )
                gen = gen if num_candidates > 1 else [gen]
                for candidate_idx, audio in enumerate(gen):
                    audio = audio.squeeze(0).cpu()
                    if candidate_idx == 0:
                        audio_parts.append(audio)
                    if output_dir:
                        filename = f"{clip_name}_{candidate_idx:02d}.wav"
                        save_gen_with_voicefix(audio, output_dir.joinpath(filename), squeeze=False, voicefixer=use_voicefixer)
                        
            audio = torch.cat(audio_parts, dim=-1)
            if output_dir:
                filename = f'{"-".join(voice)}_combined.wav'
                final_file_output = output_dir.joinpath(filename)
                save_gen_with_voicefix(
                    audio,
                    final_file_output,
                    squeeze=False,
                    voicefixer=use_voicefixer,
                )
                
        
        mp3_file_output = final_file_output.with_suffix(".mp3")
        print("Convert to mp3... {}".format(mp3_file_output))
        subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(final_file_output),
                str(mp3_file_output),
            ],
        )
        
        final_file_output.unlink()  # Delete the raw audio file
        shutil.rmtree(
            str(CUSTOM_VOICE_DIRECTORY), ignore_errors=True
        )  # Delete the custom voice dir, if it exists

        return mp3_file_output

