import torch

from oneuro.molecular.cuda_backend import CUDAMolecularBrain


def test_present_stimulus_window_counts_uses_expected_step_chunks() -> None:
    brain = CUDAMolecularBrain(4, device="cpu")

    relay_ids = torch.tensor([0, 1], dtype=torch.int64, device=brain.device)
    up_ids = torch.tensor([2], dtype=torch.int64, device=brain.device)
    down_ids = torch.tensor([3], dtype=torch.int64, device=brain.device)
    activation = torch.tensor([5.0, 7.0], device=brain.device)
    teacher_ids = torch.tensor([2], dtype=torch.int64, device=brain.device)

    step_chunks: list[int] = []

    def fake_run(steps: int) -> None:
        step_chunks.append(steps)
        brain.spike_count[2] += steps
        brain.spike_count[3] += steps * 2
        brain.external_current.zero_()

    brain.run = fake_run  # type: ignore[method-assign]

    up_count, down_count = brain.present_stimulus_window_counts(
        relay_ids=relay_ids,
        activation=activation,
        up_ids=up_ids,
        down_ids=down_ids,
        stim_steps=5,
        teacher_motor_ids=teacher_ids,
        teacher_motor_intensity=3.0,
    )

    assert step_chunks == [2, 2, 1]
    assert up_count == 5
    assert down_count == 10
    assert torch.count_nonzero(brain.external_current).item() == 0


def test_triton_cannot_be_enabled_on_cpu() -> None:
    brain = CUDAMolecularBrain(8, device="cpu")
    brain.set_triton_enabled(True)
    assert brain.triton_enabled() is False
    assert brain.triton_error() is not None
