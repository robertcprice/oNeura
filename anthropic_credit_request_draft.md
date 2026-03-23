Subject: Credit Reimbursement Request — Recurring Claude Code Session Freezes Causing Significant Credit Loss

Hi Anthropic Support,

My name is Bobby Price (bobbycprice90@gmail.com). I'm a bootstrapped developer using Claude Max for an open-source project called oNeura — a molecular-resolution terrarium/ecosystem simulation built in Rust. I love this tool and I rely on it heavily, which is exactly why I'm writing.

I'm experiencing a recurring issue where Claude Code task sessions freeze mid-execution, losing all progress and burning through credits with zero output. Today's session was a particularly costly example, but it's far from the first time.

Here's what happened today during a graphics rendering overhaul:

- **Session v1** ("Graphics overhaul continuation"): Froze at ~63 turns. Zero commits. All work lost.
- **Session v2** ("Graphics overhaul v2"): Froze at ~77 turns. Zero commits. All work lost.
- **Session v3** ("Graphics overhaul v3"): Partially survived — 2 commits made it through before freezing again with additional uncommitted work lost.
- **Session v4** ("Graphics overhaul v4"): Started to pick up where v3 left off, same freezing pattern emerging.
- **Multiple additional sessions** were required just to audit whether the frozen sessions had saved any work, adding further credit consumption with no productive output.

In total, this represents roughly 3+ hours of continuous Claude compute time — the vast majority of which produced nothing usable. The sessions show as "running" but stop advancing, don't respond to messages, and never commit their work before dying.

This is not an isolated incident. I've been experiencing this exact pattern since I started using the product in October 2025 — nearly six months now. Session freezes, lost work, and wasted credits have been a consistent issue across the entire duration of my usage, across different projects, different session types, and different times of day. Today was just a particularly painful example because of the sheer number of consecutive failures, but this has been happening regularly for months. At this point, I've lost more credits to frozen sessions than I can reasonably quantify.

As someone who is bootstrapping, every dollar of compute matters. Losing hours of credits to frozen sessions that produce no output is genuinely painful, and it's been compounding since day one.

I'm requesting a credit reimbursement for today's wasted compute at minimum. I don't have exact token counts, but the session history should be traceable on your end. Given the scope of this issue over the past six months, I'd also appreciate any broader consideration Anthropic can offer.

I also want to flag this as a product issue worth investigating — the combination of long-running code sessions that (1) don't checkpoint/commit incrementally by default and (2) are prone to freezing creates a really punishing failure mode for users. Even a simple auto-commit or progress-saving mechanism would prevent this kind of loss.

I want to keep using Claude. It's an incredible tool when it works. I just can't sustain burning credits on sessions that freeze and lose everything.

Thank you for your time,
Bobby Price
bobbycprice90@gmail.com
