# Optional API-driven training controller

This extension is outside the default server workflow. The main installation,
CLI and core tests do not require it. Its implementation is `src/vcell/agent.py`;
tests are in `optional_tests/test_agent.py`. There is no `python -m vcell agent`
alias.

Complete the fixed teacher–student experiments before exploring this controller:

```bash
python -m pip install -e '.[agent]'
python -m vcell.agent --run runs/real_01 --model YOUR_API_MODEL --max-trials 3
```

Supply `OPENAI_API_KEY` in the environment; never commit it. The model must support
the Responses API with Structured Outputs. The controller only reads its
validation summaries and proposes learning-rate, weight-decay, distillation,
mutual, contrastive and teacher-mixture settings. Architectures, seeds, data
splits and epoch limits remain fixed. Teacher predictions are reused, and students
restart from fixed initialization for each trial. Validation metrics select the
winner; generated text cannot override the metrics. Add `--resume` after an
interruption of the same experiment.

Opt-in offline tests:

```bash
python -m pip install -e '.[dev,agent]'
python -m pytest -q optional_tests/test_agent.py
```

These tests are separate from server CPU CI. Historical offline tests do not
verify live API calls, current model availability, or research benefit.
