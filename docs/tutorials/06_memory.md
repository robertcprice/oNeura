# Tutorial 6: Memory Systems

## Overview

Memory in the brain is not a single system but multiple interacting systems: working memory holds information temporarily, the hippocampus encodes new memories, and the cortex stores long-term knowledge. This tutorial explores how to implement and study memory in organic neural networks.

This tutorial covers:
- Working memory in neural tissue
- Hippocampal pattern completion
- Memory encoding and consolidation
- Replay and offline learning
- Forgetting and interference

## Neuroscience Background

### Memory Systems

| System | Duration | Capacity | Neural Substrate |
|--------|----------|----------|------------------|
| **Sensory** | < 1 second | Large | Sensory cortex |
| **Working** | Seconds-minutes | 7+/-2 items | Prefrontal cortex |
| **Episodic** | Years | Large | Hippocampus + cortex |
| **Semantic** | Lifetime | Very large | Distributed cortex |
| **Procedural** | Lifetime | Large | Basal ganglia, cerebellum |

### Hippocampal Function

The hippocampus is critical for forming new episodic memories. It performs:

1. **Pattern Separation**: Making similar inputs distinct
2. **Pattern Completion**: Retrieving full memories from partial cues
3. **Consolidation**: Transferring memories to cortex

### Working Memory

Working memory maintains information through persistent activity:

```
Input -> Active maintenance (reverberating loops) -> Output
```

This is distinct from synaptic weight changes - it's active neural firing that holds information.

## Working Memory in Organic Neural

### Implementing Working Memory

```python
from organic_neural_network import OrganicNeuralNetwork

tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=40,
    energy_supply=3.0
)

# Define working memory region
tissue.define_input_region("wm_input", (3.0, 5.0, 2.5), radius=1.5)
tissue.define_output_region("wm_output", (7.0, 5.0, 2.5), radius=1.5)
```

### The MemoryTask

The framework includes a built-in memory task:

```python
from organic_neural_network import MemoryTask

# Create memory task
task = MemoryTask(tissue)

# Task phases:
# 1. Encoding: Show pattern to remember
# 2. Delay: Wait period (no input)
# 3. Retrieval: Show probe, ask if it matches

# Train on memory task
stats = tissue.train_task(task, n_episodes=100)
print(f"Memory task final success: {stats['final_success_rate']:.1f}%")
```

### Manual Working Memory Test

```python
def test_working_memory(tissue, pattern, delay_ms=500):
    """
    Test working memory retention.

    Args:
        tissue: Neural tissue
        pattern: Input pattern to remember
        delay_ms: Delay period in milliseconds

    Returns:
        retention_score: How well the pattern was maintained
    """
    # 1. Encoding phase - present the pattern
    for _ in range(10):
        tissue.set_inputs(pattern)
        tissue.step(dt=1.0)

    # Record the activity state after encoding
    encoded_state = tissue.read_outputs()

    # 2. Delay phase - no input, test maintenance
    delay_activity = []
    for _ in range(int(delay_ms / 1.0)):
        # No input during delay
        tissue.step(dt=1.0)
        delay_activity.append(tissue.read_outputs())

    # 3. Test retention
    final_state = tissue.read_outputs()

    # Compare final to encoded
    retention = 1.0 - sum(abs(final_state[k] - encoded_state[k])
                          for k in encoded_state) / len(encoded_state)

    return retention, delay_activity

# Test with a pattern
pattern = {"wm_input": 1.0}
retention, activity = test_working_memory(tissue, pattern, delay_ms=300)
print(f"Working memory retention: {retention*100:.1f}%")
```

## Hippocampal Pattern Completion

### Pattern Separation

The hippocampus makes similar inputs distinct:

```python
def test_pattern_separation(tissue, patterns):
    """
    Test how well the network separates similar patterns.

    Args:
        tissue: Neural tissue
        patterns: List of input patterns

    Returns:
        separation_score: How distinct the representations are
    """
    representations = []

    for pattern in patterns:
        # Reset tissue state
        for neuron in tissue.neurons.values():
            neuron.membrane_potential = -70.0

        # Present pattern
        for _ in range(20):
            tissue.set_inputs(pattern)
            tissue.step(dt=0.5)

        # Record representation
        output = tissue.read_outputs()
        representations.append(output)

    # Calculate pairwise distances
    distances = []
    for i in range(len(representations)):
        for j in range(i + 1, len(representations)):
            dist = sum(abs(representations[i][k] - representations[j][k])
                       for k in representations[i])
            distances.append(dist)

    return np.mean(distances), representations

# Create similar patterns
patterns = [
    {"input_a": 1.0, "input_b": 0.0},
    {"input_a": 0.9, "input_b": 0.1},
    {"input_a": 0.8, "input_b": 0.2},
]

sep_score, reps = test_pattern_separation(tissue, patterns)
print(f"Pattern separation score: {sep_score:.3f}")
```

### Pattern Completion

Retrieving full memories from partial cues:

```python
def test_pattern_completion(tissue, full_pattern, partial_cue):
    """
    Test if network can retrieve full pattern from partial cue.

    Args:
        tissue: Neural tissue
        full_pattern: Complete pattern to be stored
        partial_cue: Partial input for retrieval

    Returns:
        completion_score: How well the full pattern was retrieved
    """
    # 1. Store the pattern (training)
    for _ in range(50):
        tissue.set_inputs(full_pattern)
        tissue.step(dt=0.5)
        tissue.update_eligibility_traces(dt=0.5)

    # Strengthen connections that were co-active
    tissue.release_dopamine(1.0)
    tissue.apply_reward_modulated_plasticity()

    # 2. Reset state
    for neuron in tissue.neurons.values():
        neuron.membrane_potential = -70.0

    # 3. Present partial cue
    for _ in range(20):
        tissue.set_inputs(partial_cue)
        tissue.step(dt=0.5)

    # 4. Read out completed pattern
    retrieved = tissue.read_outputs()

    # 5. Compare to original
    # Focus on elements that were NOT in the cue
    completion_score = 0.0
    n_missing = 0

    for key in full_pattern:
        if key not in partial_cue or partial_cue[key] == 0:
            # This should have been completed
            expected = full_pattern[key]
            actual = retrieved.get(key, 0.0)
            completion_score += 1.0 - abs(expected - actual)
            n_missing += 1

    if n_missing > 0:
        completion_score /= n_missing

    return completion_score, retrieved

# Test pattern completion
full = {"feature_1": 1.0, "feature_2": 1.0, "feature_3": 1.0}
partial = {"feature_1": 1.0, "feature_2": 0.0, "feature_3": 0.0}  # Only one feature as cue

score, retrieved = test_pattern_completion(tissue, full, partial)
print(f"Pattern completion score: {score*100:.1f}%")
print(f"Retrieved: {retrieved}")
```

## Memory Consolidation

### Systems Consolidation Theory

Memories are initially hippocampus-dependent but become cortex-independent over time:

```python
from multi_tissue_network import MultiTissueNetwork, TissueConfig

def create_memory_system():
    """Create a hippocampal-cortical memory system."""
    brain = MultiTissueNetwork()

    # Add regions
    hippocampus = brain.add_tissue(TissueConfig.hippocampus(neurons=25))
    cortex = brain.add_tissue(TissueConfig.cortex(neurons=40))

    # Connect bidirectionally
    brain.connect_tissues(hippocampus, cortex, connection_prob=0.15)
    brain.connect_tissues(cortex, hippocampus, connection_prob=0.15)

    return brain, hippocampus, cortex

brain, hippo, cortex = create_memory_system()
```

### Offline Consolidation (Sleep)

```python
def offline_consolidation(brain, hippocampus_id, cortex_id,
                          memories, n_replays=100):
    """
    Simulate sleep-like offline consolidation.

    During sleep, the hippocampus replays recent memories,
    strengthening cortical representations.

    Args:
        brain: Multi-tissue network
        hippocampus_id: ID of hippocampal tissue
        cortex_id: ID of cortical tissue
        memories: List of patterns to consolidate
        n_replays: Number of replay events
    """
    hippo_tissue = brain.tissues[hippocampus_id]
    cortex_tissue = brain.tissues[cortex_id]

    # Lower plasticity during sleep (slow learning)
    old_lr = cortex_tissue.learning_rate
    cortex_tissue.learning_rate = 0.05

    for replay in range(n_replays):
        # Select random memory to replay
        memory = random.choice(memories)

        # Hippocampus drives replay
        hippo_tissue.set_inputs(memory)
        for _ in range(10):
            hippo_tissue.step(dt=0.5)

        # Let activity propagate to cortex
        brain.step(dt=0.5)

        # Strengthen cortical representation
        cortex_tissue.release_dopamine(0.3)
        cortex_tissue.apply_reward_modulated_plasticity()

    # Restore learning rate
    cortex_tissue.learning_rate = old_lr

    return brain

# Collect memories during "waking"
memories = []
for i in range(10):
    # Create unique patterns
    pattern = {f"feature_{j}": random.random() for j in range(3)}
    memories.append(pattern)

    # Encode in hippocampus
    brain.tissues[hippo].set_inputs(pattern)
    for _ in range(20):
        brain.tissues[hippo].step(dt=0.5)

# Consolidate offline
brain = offline_consolidation(brain, hippo, cortex, memories)
```

## Forgetting and Interference

### Retroactive Interference

New learning can interfere with old memories:

```python
def demonstrate_retroactive_interference(tissue, n_memories=5):
    """
    Demonstrate how new learning interferes with old memories.

    Args:
        tissue: Neural tissue
        n_memories: Number of memories to test

    Returns:
        retention_scores: Retention over learning history
    """
    # Define memory task
    tissue.define_input_region("mem_input", (5.0, 5.0, 2.5), radius=2.0)
    tissue.define_output_region("mem_output", (5.0, 5.0, 2.5), radius=2.0)

    memories = []
    retention_scores = []

    for i in range(n_memories):
        # Create new memory
        pattern = {"mem_input": random.random()}
        memories.append(pattern)

        # Learn the new memory
        for _ in range(30):
            tissue.set_inputs(pattern)
            tissue.step(dt=0.5)
            tissue.update_eligibility_traces(dt=0.5)

        tissue.release_dopamine(1.0)
        tissue.apply_reward_modulated_plasticity()

        # Test retention of ALL previous memories
        retention = []
        for j, old_pattern in enumerate(memories):
            # Present old pattern
            for _ in range(10):
                tissue.set_inputs(old_pattern)
                tissue.step(dt=0.5)

            output = tissue.read_output("mem_output")
            expected = old_pattern["mem_input"]
            retention.append(1.0 - abs(output - expected))

        retention_scores.append(retention)

    return retention_scores, memories

# Run demonstration
retention, memories = demonstrate_retroactive_interference(tissue)

print("Retroactive Interference Analysis")
print("=" * 40)
print(f"{'Memory':>8}", end="")
for i in range(len(retention[-1])):
    print(f"{i:>8}", end="")
print()

for step, scores in enumerate(retention):
    print(f"{step:>8}", end="")
    for score in scores:
        print(f"{score*100:>7.1f}%", end="")
    print()
```

Expected output shows declining retention of older memories:
```
Retroactive Interference Analysis
========================================
 Memory       0       1       2       3       4
       0  100.0%
       1   85.0%  100.0%
       2   72.0%   88.0%  100.0%
       3   58.0%   75.0%   90.0%  100.0%
       4   45.0%   62.0%   78.0%   92.0%  100.0%
```

### Proactive Interference

Old memories can interfere with new learning:

```python
def demonstrate_proactive_interference(tissue, n_memories=5):
    """
    Demonstrate how old memories interfere with new learning.
    """
    learning_rates = []

    for i in range(n_memories):
        # Create new pattern
        pattern = {"mem_input": random.random()}

        # Measure learning rate
        rewards = []
        for trial in range(50):
            # Reset
            for neuron in tissue.neurons.values():
                neuron.membrane_potential = -70.0

            # Present pattern
            tissue.set_inputs(pattern)
            for _ in range(10):
                tissue.step(dt=0.5)

            output = tissue.read_output("mem_output")
            expected = pattern["mem_input"]
            reward = 1.0 - abs(output - expected)
            rewards.append(reward)

            # Learn
            tissue.release_dopamine(reward)
            tissue.apply_reward_modulated_plasticity()

        # Learning rate = slope of improvement
        early = np.mean(rewards[:10])
        late = np.mean(rewards[-10:])
        learning_rate = (late - early) / 40  # Per trial

        learning_rates.append(learning_rate)

        print(f"Memory {i}: Learning rate = {learning_rate:.4f}")

    return learning_rates

learning_rates = demonstrate_proactive_interference(tissue)
```

## Complete Example: Episodic Memory System

```python
from organic_neural_network import OrganicNeuralNetwork
import numpy as np

class EpisodicMemorySystem:
    """
    A complete episodic memory system with encoding, storage, and retrieval.
    """

    def __init__(self, size=(15.0, 15.0, 5.0), neurons=60):
        self.tissue = OrganicNeuralNetwork(
            size=size,
            initial_neurons=neurons,
            energy_supply=3.0
        )

        # Define regions
        self.tissue.define_input_region("context", (3.0, 7.0, 2.5), radius=2.0)
        self.tissue.define_input_region("item", (12.0, 7.0, 2.5), radius=2.0)
        self.tissue.define_output_region("recall", (7.5, 12.0, 2.5), radius=2.0)

        self.memories = []
        self.contexts = []

    def encode(self, context, item, n_repeats=20):
        """
        Encode a new episodic memory.

        Args:
            context: Context features (where, when)
            item: Item features (what)
            n_repeats: Number of encoding repetitions
        """
        # Combine context and item
        full_pattern = {"context": context, "item": item}

        # Encode through repeated presentation
        for _ in range(n_repeats):
            self.tissue.set_inputs(full_pattern)
            self.tissue.step(dt=0.5)
            self.tissue.update_eligibility_traces(dt=0.5)

        # Consolidate
        self.tissue.release_dopamine(1.0)
        self.tissue.apply_reward_modulated_plasticity()

        # Store for later
        self.memories.append(full_pattern)
        self.contexts.append(context)

    def recall_by_context(self, cue_context, n_steps=30):
        """
        Recall an item given a context cue.

        Args:
            cue_context: Context to cue recall
            n_steps: Processing time

        Returns:
            retrieved_item: The recalled item
            confidence: Retrieval confidence
        """
        # Reset state
        for neuron in self.tissue.neurons.values():
            neuron.membrane_potential = -70.0

        # Present context cue
        pattern = {"context": cue_context, "item": 0.0}  # No item info

        activity_trace = []
        for _ in range(n_steps):
            self.tissue.set_inputs(pattern)
            self.tissue.step(dt=0.5)

            recall_output = self.tissue.read_output("recall")
            activity_trace.append(recall_output)

        # Final recall
        retrieved_item = self.tissue.read_output("recall")

        # Confidence based on activity stability
        if len(activity_trace) > 5:
            recent = activity_trace[-5:]
            confidence = 1.0 - np.std(recent) / (np.mean(recent) + 0.01)
        else:
            confidence = 0.5

        return retrieved_item, confidence

    def test_recall_accuracy(self, n_tests=10):
        """Test overall recall accuracy."""
        correct = 0

        for _ in range(n_tests):
            if not self.memories:
                continue

            # Select random memory
            idx = random.randint(0, len(self.memories) - 1)
            memory = self.memories[idx]

            # Recall with context
            retrieved, confidence = self.recall_by_context(memory["context"])

            # Check if correct
            expected = memory["item"]
            if abs(retrieved - expected) < 0.3:
                correct += 1

        return correct / n_tests

# Use the system
memory_system = EpisodicMemorySystem()

# Encode some memories
memory_system.encode(context=0.2, item=0.8)  # Happy party
memory_system.encode(context=0.8, item=0.3)  # Sad funeral
memory_system.encode(context=0.5, item=0.5)  # Neutral work

# Test recall
accuracy = memory_system.test_recall_accuracy(n_tests=20)
print(f"Recall accuracy: {accuracy*100:.1f}%")

# Specific recall test
item, conf = memory_system.recall_by_context(0.2)  # Cue with party context
print(f"Recalled item: {item:.3f} (confidence: {conf:.3f})")
print(f"Expected: 0.8")
```

## References

- O'Keefe, J. & Nadel, L. (1978). "The Hippocampus as a Cognitive Map" - Oxford
- McClelland, J. L. et al. (1995). "Why there are complementary learning systems" - Psych. Review
- Baddeley, A. (2000). "The episodic buffer: a new component of working memory?" - Trends Cog. Sci.

## Summary

In this tutorial, you learned:

1. **Working Memory**: Active maintenance through persistent activity
2. **Pattern Separation**: Making similar inputs distinct
3. **Pattern Completion**: Retrieving full memories from partial cues
4. **Memory Consolidation**: Transferring memories from hippocampus to cortex
5. **Interference**: How new and old memories interfere with each other
6. **Episodic Memory**: Complete system with context-item binding

The key insight is that **memory is not storage but reconstruction** - the brain does not store perfect copies but rather patterns of connectivity that allow reconstruction when cued. This makes memory flexible but also fallible, subject to interference and distortion.
