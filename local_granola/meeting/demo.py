from __future__ import annotations

from pathlib import Path

from .pipeline import generate_transcript_artifacts


def main() -> None:
    segments = [
        {"start": 1.2, "end": 4.8, "text": "Let's start with the weekly launch review."},
        {"start": 8.4, "end": 12.1, "text": "Friday still looks realistic from engineering."},
        {"start": 18.0, "end": 23.5, "text": "We may need to repeat the deployment checklist on the call."},
        {"start": 31.0, "end": 35.2, "text": "Marketing assets are the remaining blocker."},
        {"start": 43.0, "end": 46.6, "text": "Sarah will send the revised assets tomorrow."},
        {"start": 63.3, "end": 66.7, "text": "Decision confirmed, we launch on Friday."},
    ]

    artifacts = generate_transcript_artifacts(
        segments=segments,
        output_directory=Path("output") / "demo_transcript",
        source_audio_path="output/demo_transcript/mixed.wav",
    )
    print(artifacts)


if __name__ == "__main__":
    main()
