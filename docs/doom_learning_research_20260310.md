# Doom Learning Research Notes

Date: 2026-03-10

## Bottom line

The current system is learning, but mostly through coarse local shaping:

- `health_gathering`: real improvement
- `defend_the_center`: real kill improvement after relay-targeted credit assignment
- `deadly_corridor`: better positioning and survival, but not reliable kill improvement

The main reason is not "the brain cannot learn." The main reason is that the current Doom loop is using a weak and blunt learning interface:

1. reward is sparse and often arrives too late
2. the action decoder is mostly fixed
3. teacher pretraining is mostly behavior cloning without online dataset correction
4. the demo often bypasses the backend's built-in three-factor mechanism with direct synapse nudges

## What the literature suggests

### 1. Three-factor learning is the right biological template, but it needs eligibility traces

Frémaux and Gerstner argue that reward-modulated learning needs:

- presynaptic activity
- postsynaptic activity
- a third factor such as dopamine or another modulatory signal

Critically, the synapse needs an eligibility trace so delayed reward can still reinforce the causal synapses.

Implication for oNeuro:

- this matches what the backend already implements
- the Doom demo should lean harder on eligibility + dopamine instead of mostly hand-editing `syn_strength`

### 2. The backend already contains a stronger rule than the Doom demo is using

`src/oneuro/molecular/cuda_backend.py` already has:

- `syn_eligibility`
- slow eligibility decay
- dopamine-gated conversion of eligibility into permanent weight change

This is much closer to what the literature recommends than the current demo's manual relay-to-motor Hebbian nudge.

Implication:

- the first high-value change is not inventing a new learning rule
- it is routing Doom reward through the existing three-factor backend more directly

### 3. DishBrain used a very simple interface and a very simple task

Kagan et al. worked because the task was simple and the interface was tightly constrained:

- low-dimensional sensory input
- low-dimensional action output
- immediate closed-loop feedback
- predictable vs unpredictable stimulation

Implication:

- Doom is much harder than Pong
- if we want DishBrain-style learning to work in Doom, we need a similarly disciplined interface
- the current movement/turn/strafe/attack setup is still too unconstrained for sparse biological-style reward alone

### 4. Teacher forcing alone is not enough; interactive imitation matters

Ross et al. (DAgger) is directly relevant here.

Why:

- pure teacher-forced pretraining learns on teacher-induced states
- free-play produces different states
- errors compound because the learner drifts off the teacher distribution

Implication:

- the current teacher pretraining in Doom should evolve into DAgger-style relabeling on the learner's own trajectories
- this is especially important for `deadly_corridor`, where one bad turn or attack changes future observations a lot

### 5. Spiking control systems work better when the readout is trainable

Bellec et al. (`e-prop`) and the surrogate-gradient literature both point in the same direction:

- keep the recurrent/spiking substrate
- train the output/readout layer explicitly
- use local eligibility traces in the recurrent substrate
- let a stronger optimization signal shape the readout

Implication:

- a fixed winner-take-all decoder is too restrictive
- the fastest path to better Doom learning is likely: keep the brain as the recurrent substrate, but train a lightweight readout on top of it

## What is most likely wrong in the current Doom stack

### A. Reward credit is too blunt

Current behavior:

- reward often lands on "whatever was active recently"
- manual relay-to-motor nudges are broad and hand-designed
- kill reward is still sparse relative to the action horizon

Why this hurts:

- `deadly_corridor` needs multi-step credit assignment
- positioning, alignment, and firing all contribute
- a kill-only reward arrives too late unless eligibility is doing the real work

### B. The decoder is a bottleneck

Current behavior:

- the decoder is mostly a spike-count argmax
- recent decoder-bias experiments improved corridor survival and attack-window firing, which is a strong sign that readout selection is limiting performance

Why this matters:

- if a small decoder prior changes behavior this much, the readout is underpowered
- this is a good sign: it means the recurrent substrate is probably producing usable state information already

### C. Combat is really two tasks, not one

Combat requires at least two separable subproblems:

- movement/positioning
- attack timing

Current behavior:

- the corridor teacher improved survival and damage first
- attack-window firing remained the weak point

Implication:

- the policy should probably have separate movement and attack heads
- trying to learn both with one flat winner-take-all readout wastes sample efficiency

### D. Teacher pretraining is still open-loop

Current behavior:

- pretraining uses teacher actions
- then free play starts
- the teacher is not systematically queried on learner-generated states during training

Implication:

- this is classic covariate shift
- DAgger-style relabeling should help more than longer pure pretraining

## Ranked roadmap

### Tier 1: highest expected return

#### 1. Switch Doom learning to backend eligibility + dopamine

Goal:

- stop relying primarily on `_doom_hebbian_nudge`
- let `syn_eligibility` and DA-gated plasticity do the heavy lifting

Concrete change:

- during play, record which relay subset and motor subset were active
- deliver dopamine to the relevant postsynaptic populations on reward
- optionally deliver serotonin/stress on damage
- remove or greatly reduce manual direct weight edits

Why this is first:

- the backend already supports it
- it is more biologically grounded
- it is better suited for delayed reward than direct edits

Success criterion:

- `defend_the_center` mean kills should stay at least as good as the current active-relay baseline
- `deadly_corridor` should improve mean kills without sacrificing survival

#### 2. Replace the fixed decoder with a trainable readout

Goal:

- keep the molecular brain as the recurrent state engine
- train the readout explicitly

Concrete change:

- use L5 spike counts or short spike-history features as input
- train a small linear or 2-layer readout
- start with behavior cloning / DAgger
- then fine-tune with PPO or actor-critic

Why this is second:

- current experiments already show decoder gating changes behavior a lot
- this is the strongest sign that trainable readout will help

Success criterion:

- attack-window fire rate rises
- blind attacks fall
- `deadly_corridor` mean kills goes up on held-out seeds

#### 3. Turn teacher pretraining into DAgger

Goal:

- fix covariate shift from pure teacher forcing

Concrete change:

- run the learner policy
- query the teacher on visited states
- aggregate `(brain state, teacher action)` data
- retrain the readout repeatedly

Why this matters:

- Doom is sequential and highly state-distribution sensitive
- DAgger was built for exactly this failure mode

Success criterion:

- teacher match on free-play trajectories improves
- `deadly_corridor` no longer collapses after the learner drifts off-distribution

### Tier 2: likely useful after Tier 1

#### 4. Split movement and attack decoding

Concrete change:

- movement head: forward / turn / strafe
- attack head: fire / no-fire
- combine them into final actions

Reason:

- attack timing and navigation are partially separable
- current flat 6-way argmax makes the learner trade off movement vs firing in a brittle way

#### 5. Add curriculum by behavior, not just by scenario

Recommended sequence:

1. Pong `retina_body + teacher + replay`
2. Doom `health_gathering`
3. Doom `defend_the_center`
4. Doom `deadly_corridor`

But also split by skill:

1. center enemy
2. fire when centered
3. survive while centering
4. combine into full combat

#### 6. Use a tiny trainable observation adapter before the brain

Reason:

- DishBrain worked with a very controlled interface
- your visual input is still much noisier

Concrete change:

- a small learned adapter maps retina output to relay currents
- keep the adapter low-capacity so the brain still matters

## Experiments to run next

### Experiment A: three-factor Doom

Compare on `defend_the_center` and `deadly_corridor`:

- current active-relay baseline
- backend-eligibility + dopamine
- backend-eligibility + dopamine + DAgger readout

Metrics:

- kills
- survival
- damage
- attack-window fire rate
- blind attack shots

### Experiment B: trainable readout only

Freeze recurrent brain plasticity.

Train only:

- linear readout
- or tiny MLP readout

If this gives a big jump, the bottleneck is decoder capacity, not substrate dynamics.

### Experiment C: DAgger vs pure teacher pretraining

Compare:

- teacher pretraining only
- teacher pretraining + DAgger

This should be run before more elaborate biological changes, because it directly tests the current failure mode.

## Recommendation

If the goal is "make it learn better quickly" rather than "stay maximally pure," the best sequence is:

1. use backend eligibility + dopamine in Doom
2. add a trainable decoder
3. train that decoder with DAgger
4. only then revisit more biologically elaborate reward shaping

If the goal is "make the biological mechanism itself stronger," then:

1. remove most direct hand-edit nudges
2. rely on eligibility traces and modulators
3. keep the decoder simple, but still trainable

## Sources

- Kagan et al., 2022, *In vitro neurons learn and exhibit sentience when embodied in a simulated game-world*:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9747182/
- Frémaux and Gerstner, 2016, *Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules*:
  https://www.frontiersin.org/articles/10.3389/fncir.2015.00085/full
- Gerstner et al., 2018, *Eligibility Traces and Plasticity on Behavioral Time Scales*:
  https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00053/full
- Bellec et al., 2020, *A solution to the learning dilemma for recurrent networks of spiking neurons*:
  https://www.nature.com/articles/s41467-020-17236-y
- Neftci, Mostafa, and Zenke, 2019, *Surrogate Gradient Learning in Spiking Neural Networks*:
  https://par.nsf.gov/biblio/10133050-surrogate-gradient-learning-spiking-neural-networks-bringing-power-gradient-based-optimization-spiking-neural-networks
- Ross, Gordon, and Bagnell, 2011, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*:
  https://proceedings.mlr.press/v15/ross11a.html
