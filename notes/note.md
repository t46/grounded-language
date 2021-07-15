# About this project
- I thinks it matters that agents realize that humans attatch symbols to objects in this world.
  - I would like to create such agents as a first step for creating a general artificial intelligence.
- A work, *[Grounded Language Learning Fast and Slow](https://arxiv.org/abs/2009.01719)*, by Felix Hill et al. is a promising research in this direction.
- Running through the research, I come into the hypothesis that rich representation of our world matters for fast mapping of language.
  - The reasons are as follows
    - Adding reconstruction loss seems to dramatically improve the performance
    - Authors point out that *temporal aspect*, observing an object from multiple views, matters
  - This is also consistent with my belief that Helen Keller's language acquision is supported by her rich world representation.
- Thus, I plan to test this hypothesis by a seriese of experiments.

# Rough Implementation Plan
- Authors provide task environment as a docker image.
  - However, I find it tough to do something in the container built from the image.
    - Thus, it may be tough to modify tasks.
- Authors do not provide codes of agents
  - Thus, I have to write codes of reinforcement learning algorithms.
- Given them, I'm going to proceed this project in the following order.
    1. I first try to run experiments in the docker image by a random agent provided in the repository.
    2. I implement a simple base agent that plays in the environment.
    3. I implement the recunstruction part.
    4. I implement the memory part.
    5. I modify the network and test my hypothesis by comparing them.