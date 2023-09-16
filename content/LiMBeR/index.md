---
emoji: 📄
title: Linearly Mapping from Image to Text Space
date: '2023-09-16 20:00:00'
author: hagyeong
tags: PaperReview Multi-Modal
categories: PaperReview Multi-Modal
---
## EffL LAB. Regular Seminar 

# Linearly Mapping from Image to Text Space (ICLR'23)

![](./imagebundle/LiMBeR_01.png)


## Problem of Language Model 
![](./imagebundle/LiMBeR_02.png)

Emily M. Bender and Alexander Koller., “Climbing towards NLU: on meaning form and understanding in the age of data”, ACL 2020

A System exposed only to form in its training cannot in principle learn meaning

##Form & Meaning in Language** 
![](./imagebundle/LiMBeR_03.png)
Form 

- Anything we can find in a language (e.g., symbols, mouth movements)

Meaning

- Relationship between form and non-linguistic parts
- Including Communicative intent

Is **form** alone **meaningful?** 

## Octopus Thought exp.
![](./imagebundle/LiMBeR_04.png)
A highly intelligent octopus that knows nothing about Human language

- Excellent at spotting *statistical* patterns

- Observed the use of certain words in similar **forms**
- Maybe noticed a common lexical pattern
![](./imagebundle/LiMBeR_05.png)
![](./imagebundle/LiMBeR_06.png)
starts impersonating B and replying to A
![](./imagebundle/LiMBeR_07.png)
**The octopus doesn't know the referents of the words **no idea what bears or sticks are**
- => **Octopus = LM**

## Octopus Thought Experiment - Conclusion
![](./imagebundle/LiMBeR_08.png)
- LMs do not tend to learn conceptual representations (meanings) of language.
- Humans acquire language not only through the **form** (representation) 

but also through the **interaction** of various factors in physical world.

***How well can a text-only language model learn aspects of the physical world?**


## Previous Works
![](./imagebundle/LiMBeR_09.png)
- Show success in mapping images to language model soft prompts as a method for multimodal pre-training (e.g., *MAGMA*, *Frozen*)
    - Constantin Eichenberg et al., “MAGMA–Multi modal Augmentation of Generative Models through Adapter-based Finetuning”, EMNLP 2022

    -  Maria Tsimpoukelli et al., “Multimodal Few-Shot Learning with Frozen Language Models”, NeurIPS 2021

- However, no attempts to restrict the mechanism behind this mapping and understand how it works.

## Language & Image representation
![](./imagebundle/LiMBeR_10.png)
- **Hypothesis.**

Conceptual representations (between language and image embeddings) can be approximately mapped to one through a linear transformation

- Why train on linear transformation?
- because of the simplicity !


## Method
![](./imagebundle/LiMBeR_11.png)
LiMBeR (Linearly Mapping Between Representation spaces)

- Train linear projections from image representations into the text space of a language model to produce image-to-text tasks

= transform an image representation into “soft prompts” 

(do not correspond to discrete language tokens)

![](./imagebundle/LiMBeR_12.png)
![](./imagebundle/LiMBeR_13.png)
![](./imagebundle/LiMBeR_14.png)
![](./imagebundle/LiMBeR_15.png)
![](./imagebundle/LiMBeR_16.png)
![](./imagebundle/LiMBeR_17.png)
![](./imagebundle/LiMBeR_18.png)
![](./imagebundle/LiMBeR_19.png)
## Experiments : Captioning
![](./imagebundle/LiMBeR_20.png)
![](./imagebundle/LiMBeR_21.png)
## Experiments : VQA (Visual Question Answering)
![](./imagebundle/LiMBeR_22.png)
![](./imagebundle/LiMBeR_23.png)
## Experiments : Visual Concepts
![](./imagebundle/LiMBeR_24.png)
**Why BEIT prompts perform so poorly for VQA despite performing decently for captioning?**
![](./imagebundle/LiMBeR_25.png)

- **Hypothesis.** BEIT does not encode visual info. that corresponds to lexical categories
- Metrics
- Wu-Palmer similarity
- Calculate the distance between the GT and the generated word in the WordNet taxonomy
- Measure **how close** a word was to the correct answer


## Conclusion
![](./imagebundle/LiMBeR_26.png)
- Show the linguistic supervision of the vision model pretraining objective correlates with the degree of similarity
  - Verified a hypothesis : training only a linear layer is enough for mapping visual pre-trained knowledge to text space.
  - And it can enable downstream tasks (such as few/zero-shot VQA, image captioning) utilizing stored knowledge from both worlds
- Future work (or Question)
- Could it be improved by considering different model sizes ? 

(e.g. larger or smaller CLIP models or supervised resnets or BEITs)

- whether the probing results get better or worse with image encoder size 

```toc
```