# Sample expected outputs

These are *reference* outputs — what we expect the model (Gemini) to produce when given
the three mock scenarios in `notebooks/prompt_exploration.py`. Use them as a
quality bar when iterating on the prompt template.

If your real model responses are noticeably worse than these
(rambling, jargon-laden, vague, missing timestamps), the prompt needs
tightening. If they're noticeably different in *style* but equally good,
that's fine — these aren't ground truth, just a target.

---

## Scenario: `defending_phase` (English)

**Input:** Team A defending in own half. Compactness volatile, two events:
compression at t=4.2s (-34%), stretch at t=8.7s (+41%).

```json
{
  "headline": "Your team stayed compact most of the clip but lost shape twice — first squeezing far too tight at 4.2s, then stretching out by 41% just four seconds later.",
  "implication": "When defenders and midfielders bunch around the ball at 4.2s, the far side of the pitch is unguarded — that's why the team had to scramble back into shape at 8.7s and ended up stretched. The whole back line was reacting to one player rather than moving together.",
  "coaching_cue": "Shift as a unit, don't chase the ball — keep about ten paces between you and the next defender."
}
```

What to check:
- Names a *specific* timestamp from the events list ✓
- Connects the two events into one story (compression caused the stretch) ✓
- Coaching cue is something you'd actually shout, ≤ 12 words ✓
- No jargon ✓

---

## Scenario: `stable_block` (English)

**Input:** Team A holds a tight mid-block for 12 seconds. No events detected.

```json
{
  "headline": "Your team held a tight, stable defensive block for the full 12 seconds — no big shape changes, everyone moved together.",
  "implication": "When the team stays compact like this, the opposition has to risk longer passes around your block instead of getting through it. This is exactly what a mid-block is supposed to do.",
  "coaching_cue": "Whatever you said before this clip, keep saying it — this is the shape you want."
}
```

What to check:
- Doesn't invent drama when nothing happened ✓
- Validates good behaviour rather than searching for a problem ✓
- Coaching cue acknowledges that no change is needed ✓

---

## Scenario: `stretched_attack` (English)

**Input:** Team A attacking, gets too stretched. Two stretch events at t=3.5s
(+38%) and t=9.8s (+31%) against a compact Team B mid-block.

```json
{
  "headline": "When attacking, your team got too stretched out — the front and back of the team were 32% further apart than they should be, and twice ran way past Team B's compact block at 3.5s and 9.8s.",
  "implication": "If you lose the ball with the team that spread out, there's a huge gap between your attackers and defenders for Team B to counter through. The full-backs especially are pushing forward without the centre-backs sliding up to support.",
  "coaching_cue": "When you push up to attack, stay connected — back four steps up together, no one ahead of the ball alone."
}
```

What to check:
- Identifies that the issue is attacking, not defending ✓
- Specific risk articulated (counter-attack vulnerability) ✓
- Coaching cue addresses the actual cause (back line not stepping up) ✓

---

## Quick test — Thai output

Same `defending_phase` scenario, `lang="th"`:

```json
{
  "headline": "ทีมของคุณรักษารูปทรงได้ดีตลอดคลิป แต่เสียทรงสองครั้ง — ตอน 4.2 วินาทีหดตัวมากเกินไป แล้วอีก 4 วินาทีต่อมาก็ยืดออกถึง 41%",
  "implication": "ตอนที่กองหลังกับมิดฟิลด์มากระจุกรวมกันใกล้บอลที่ 4.2 วิ ฝั่งตรงข้ามของสนามไม่มีคนคุม ทีมเลยต้องวิ่งกลับมาตั้งทรงใหม่ที่ 8.7 วิ จนเสียระยะห่างไป — คือเล่นตามบอล ไม่ได้เคลื่อนเป็นทีม",
  "implication_alt": "(แปล) When defenders and midfielders bunch around the ball at 4.2s...",
  "coaching_cue": "เคลื่อนเป็นแนวเดียวกัน อย่าวิ่งไล่บอลเดี่ยว — ห่างคนข้างๆ ประมาณ 10 ก้าวพอ"
}
```

What to check:
- Uses Thai natural register, not academic Thai ✓
- "เคลื่อนเป็นแนวเดียวกัน" = "shift as a unit", which is what a real Thai coach says ✓
- Numbers and timestamps preserved exactly ✓
