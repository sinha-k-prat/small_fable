# Discovered prompts — what actually unlocks a frozen Qwen2.5-1.5B-Instruct

Verified locally on CPU (the cached model), graded with the HARD_BENCH **strict** grader.
Only prompts with a recorded, reproduced result go here. WIP — added as they're confirmed.

---

## multidigit_multiplication — PARTIAL (decomposition works; final addition slips)

**Prompt (generic, no problem-specific content):**
- *system:* "You are a meticulous step-by-step solver. Show ALL intermediate work explicitly on
  separate lines, never skip or summarize a step. End with a line 'FINAL ANSWER:' then only the answer."
- *user:* `<the problem>` + "Multiply by long multiplication: multiply the top number by each digit
  of the bottom number (write each partial product, shifted by place), then add the partial products."

**Result on `9246 × 897` (gold 8293662), max_new_tokens=512:**
```
9246 × 7   = 64722     ✓
9246 × 90  = 832140    ✓
9246 × 800 = 7396800   ✓
add        → 8256722   ✗   (correct sum is 8293662)
```
**GRADE = 0.0**, but a NEAR MISS: all three partial products are correct. The frozen 1.5B *can* do
the hard sub-steps (multi-digit × single-digit) when the scratchpad forces them out one line at a
time. The single remaining error is the **final multi-addend addition**, which the model did all at
once and botched.

**Lesson:** the capability wall breaks under **recursive decomposition into single operations**. The
fix is to also decompose the addition (add two numbers at a time, column by column with explicit
carries) instead of summing three at once. (Refinement under test.)
