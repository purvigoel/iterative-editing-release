## Iterative Motion Editing with Natural Language ##
(SIGGRAPH 2024)
Purvi Goel, Kuan-Chieh Wang, C. Karen Liu, Kayvon Fatahalian

Text-to-motion diffusion models can generate realistic animations from text prompts, but do not support fine-grained motion editing controls. In this paper, we present a method for using natural language to iteratively specify local edits to existing character animations, a task that is common in most computer animation workflows. Our key idea is to represent a space of motion edits using a set of kinematic motion editing operators (MEOs) whose effects on the source motion is well-aligned with user expectations. We provide an algorithm that leverages pre-existing language models to translate textual descriptions of motion edits into source code for programs that define and execute sequences of MEOs on a source animation. We execute MEOs by first translating them into keyframe constraints, and then use diffusion-based motion models to generate output motions that respect these constraints. Through a user study and quantitative evaluation, we demonstrate that our system can perform motion edits that respect the animator's editing intent, remain faithful to the original animation (it edits the original animation, but does not dramatically change it), and yield realistic character animation results.

## Intro ##

This is the official code release for "Iterative Motion Editing with Natural Language". The project was implemented within a much larger codebase, built by <a href="https://wangkua1.github.io/">Kuan-Chieh Wang</a> and <a href="https://jmhb0.github.io/">James Burgess</a>, which itself was built off of great works including <a href=https://github.com/GuyTevet/motion-diffusion-model>MDM</a>, <a href="https://github.com/mkocabas/VIBE">VIBE</a>, <a href="https://github.com/akanazawa/hmr">HMR</a>, and <a href="https://smpl.is.tue.mpg.de/">SMPL</a>.
Most of the relevant files are in the `generative_infill/` folder (see, in particular, `generative_infill/generative_infill.py`) and the root folder (see `openai_wrapper.py`).

## Getting Started ##

### Front-end (Natural Language -> MEO Programs)

The <b>fundamental</b> idea of our work is framing fine-grained animation editing as program synthesis. To create programs from natural language, run `python3 openai_wrapper.py chatbot`. This will open a lightweight chatbot in your terminal. You can input text instructions with the format `<Original motion description>. <Editing instruction>`, e.g., `The person is kicking with the right leg. Kick higher.`. The chatbot will query the LLM using our prompt structure, and print out the generated MEO program. Further iterative instructions do not require the description of the original motion (just `Even higher!` will do). 

In our implementation, we use ChatGPT as our LLM. As a result, an OpenAI key will need to be provided in ``openai_wrapper.py``. Please add the code `openai.api_key = <YOUR API KEY HERE>` right under the import statements in ``openai_wrapper.py``. Make sure your current environment has the openai library installed, e.g., ``conda install conda-forge::openai``.

### Extensions ###
Want to try adding some new MEOs? Check out `llm/prog_prompt3.py`, which contains the prompt structure we feed to the LLM. Import new MEOs at the top of the file (`import <MEO_NAME>`), add a few in-context learning examples to the bottom of the file to show the LLM how to use the MEO. Then try `python3 openai_wrapper.py chatbot` and enter an instruction that ought to target the new MEO. Please feel free to contact us if we can assist.

### Full system (Natural Language -> Edited Motions)

Coming soon! 

### Development ###

This is a research prototype and, in general, will not be regularly maintained long after release. 

### Acknowlegements ###
This work was supported by a Stanford Interdisciplinary Graduate Fellowship, the Stanford Wu Tsai Human Performance Alliance, Meta and Activision.
