# Study 1: models and repetitions (draft paper text)

Draft text for the experimental setup in Section 6.3, written 2026-11-06 after the model screen. The design it
describes: Claude Opus 5.5, GPT-6.1 Sol and GPT-6.1 Terra, all six settings, 10 runs per cell for every model. The
screen records are in `/home/jongh/recovery_review_1030/screen_20261106/SCREEN.md`.

## Draft text

**Models.** The framework was evaluated with three LLMs from two providers: Claude Opus 5.5 (Anthropic) and GPT-6.1
Sol and GPT-6.1 Terra (OpenAI). Using two providers tests whether the results depend on one model family. Opus 5.5 and
Sol are each provider's flagship model. Terra is a lower-priced variant, which tests whether the checks also help a less
capable model. All three models ran at medium reasoning effort, with identical prompts, the same 40-request limit per
attempt, and the same reply schema.

The models were selected in a screening step whose criteria were fixed before it ran. A model qualified if its replies
could be parsed, the neuro-symbolic framework (event-level) recovered at least two of the four scenarios, and no attempt
failed for reasons unrelated to the model. Five candidates were screened. Claude Sonnet 5.5 was excluded because the
framework recovered only one of the four scenarios. Claude Opus 4.6 was excluded because it is an earlier model of the
same family as Opus 5.5 and was about five times slower per request than the other candidates. Screening attempts are
not part of the reported results.

**Repetitions and analysis.** LLM replies vary between runs even with identical inputs. Each combination of model,
method and scenario was therefore run 10 times, giving 720 attempts in total (3 models × 6 settings × 4 scenarios ×
10 runs). Every model and setting received the same number of runs, so pooled results are not weighted toward any one
model. Pooled over the four scenarios, each setting has 40 attempts per model.

The design varies the failure scenario rather than adding repetitions, since generality across failures is the main
question. The four scenarios differ in the failed resource (a part, a machine, a conveyor and a robot) and in the
structure of the recovery they require.

Pass rates are reported with 95% Wilson score intervals. Methods are compared with Cochran–Mantel–Haenszel tests
stratified by scenario, with Holm correction for multiple comparisons. Comparisons whose expected difference is small
relative to this sample size are reported descriptively: outline-level against event-level within the framework, and
the baseline against the informed baseline.

**Limitations (one sentence).** All scenarios take place in one manufacturing cell, so the results show generality
across failure types in this cell, not across cells or plants.

## Notes for the authors (not paper text)

- **Reproducibility:** GPT-6.1 Sol and Terra have no dated model snapshots; their IDs (`gpt-6.1-sol`,
  `gpt-6.1-terra`) are aliases. Report the dates you accessed each model, and the model IDs (`claude-opus-5-5`,
  `gpt-6.1-sol`, `gpt-6.1-terra`).
- **No dollar costs in the paper.** Run effort goes in as tokens, turns and time.
- **If a reviewer asks why 10 runs,** the strongest answer is the size of the effect: in the screen, the three chosen
  models passed 0 of 24 baseline attempts (one of them ended in a harness error, since fixed; 0 of 40 across all five
  screened models), while the framework passed 12 of 12 (Opus 5.5 4/4 in its v7 reference runs, Sol 4/4, Terra 4/4).
  Ten runs per cell is enough to show a difference that large.
- **Screen caveats:** the screen ran event-level settings only, one run per scenario, on commit `b816b1d` and the v7
  captures. Part slippage has since been re-captured with the drop point moved to (0.15, 0.72) (commit `cc2f4ac`), so
  Study 1 uses the new Part slippage capture `capture_f97ce5528f8c4a19a54211c74beec8c0`; the other three scenarios
  keep their v7 captures.
- **Check the test names** against the analysis plan in `EXPERIMENT_DESIGN_AND_PLAN.md` once it is written, so the
  paper and the plan say the same thing.
