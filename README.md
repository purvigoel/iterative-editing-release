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

Once you have the front end of the system running, you can follow the steps below to apply the generated MEOs to an input motion sequence.

#### Stepping through the code #### 

This is a big repository, built off many different repositories. If you want to get a sense for what the system is doing, your best bet is to start with the `generative_infill` folder. This is where most of the system code lives. `generative_infill/generative_infill.py` will be the entry into the system; it starts a chat to collect user input via the terminal, sends that input to the LLM for translation into MEOs, applies the MEOs as keyframe edits to a source motion, and invokes diffusion models to solve for transition movement.

#### Setting up the full system -- here we go! ####

1. Clone the repo: `` git clone https://github.com/purvigoel/iterative-editing-release.git ``

2. Set up the conda environment snapshotted in `environment.yml` with the command `conda env reate -f environment.yml`

3. Download dummy data. This data is important for setting up dataloaders/model inputs. It consists of some motions from <a href="https://amass.is.tue.mpg.de/">AMASS</a>, e.g., ACCAD. Please follow the rules specified in the AMASS website for using their dataset, including citing their work appropriately if you use it in research publications. Ensure that you have the necessary permissions and adhere to their terms of use.
   <br>
   a) The dummy data can be downloaded at this Google Drive link: https://drive.google.com/file/d/1ju-aeHJ8hNBJrmDwG9FslA4f53Q4l5S_/view?usp=sharing
   <br>
   b) Place the data at the path `VIBE/data/vibe_db/amass_db_small.pt`

5. Download the SMPL body model. Specifically, we use `SMPL_NEUTRAL.pkl`.
   <br>
   a) You can download the model from the website https://smpl.is.tue.mpg.de/. You will have to create an account and sign in.
   <br>
   b) Place SMPL_NEUTRAL.pkl at the path `body_models/smpl/SMPL_NEUTRAL.pkl`

7. Download diffusion models. There are three models--one for unconditional generation, one for trajectory infilling, and one for full-pose infilling. You'll need all of them.
   <br>
   a) You can download the models at this Google Drive link: https://drive.google.com/drive/folders/1ZcKEssmmLopnoh2NDDpX9xoSRxYdKzcJ?usp=drive_link. As you'll see, the folder is called `meos`and contains all three models.
   <br>
   b) Place the folder in the `mdm/` directory, i.e., `mdm/meos/*`

That's all the downloads complete! Now we can move on to actually running the model. As mentioned earlier, the entry point to the system is in `generative_infill/generative_infill.py`. Have a look at where the <a href="https://github.com/purvigoel/iterative-editing-release/blob/new-branch4/generative_infill/generative_infill.py#L311">source motion is loaded</a>. We provide 3 example motions that can be used as source motions in the `asset_library/` folder and load them into a dictionary accessed as `asset_library.asset_library`. You can access the motions via their IDs, e.g., `asset_library.asset_library[ 0 ]`. You can see what the motions look like via the videos included in the `asset_library/` folder.

If you'd like to use your own motions, you'll have to convert them into our data representation. <b> Note that our data representation is different from the popular HumanML3D representation. </b> I'm including a note at the end of this section with some pointers about how to do the conversion.

1. You can run the system using the following command: `CUDA_VISIBLE_DEVICES=0 bash run_scripts/ghmr/generative_infill.sh 0`. The system will take a few moments to load. Eventually, you'll see a chat open in the terminal that should look the same as when you set up the system front-end, as `You:   `

2. Time to prompt the system. Recall that the first message to the system will require a short description of the original source motion, and an instruction. `Example: The person is jumping. At the start of the motion, raise your arms.` Click enter to run. You'll see a lot of print-outs, which includes the generated program, some motion statistics, and a lot of (poorly organized) logging. 
   
3. The output motion will be saved in the `save_dir` folder specified at <a href="https://github.com/purvigoel/iterative-editing-release/blob/new-branch4/generative_infill/generative_infill.py#L70">the top of the file</a>. Right now, it's `dump_results/`, and the output motion is saved as `<save_dir>/synth_llm<iteration_number>_iter_joints.npy`.
   <br>
   a) The motion is written out as SMPL joints, into a numpy file. The data has shape (1, 60, 22, 3): 60 frames, 22 joints, 3 XYZ world-space positions per joint.
   <br>
   b) You can use your own tools to visualize these motions (I believe the original <a href="https://github.com/GuyTevet/motion-diffusion-model">MDM repository</a> comes with one). I've also written a no-frills <a href="https://github.com/purvigoel/tiny-motion-visualizer.git">web-based skeleton visualizer</a> that handles this data and visualizes it at a localhost port. Run it with `python3 viewer.py -p <PORT> -d <PATH_TO_JOINTS_FOLDER>`. 

4. If you don't want to use the LLM front-end, and would prefer to write your own MEO programs, you can write the program in the <a href="https://github.com/purvigoel/iterative-editing-release/blob/new-branch4/generative_infill/generative_infill.py#L296"> `execute` string </a>. Then <a href="https://github.com/purvigoel/iterative-editing-release/blob/new-branch4/generative_infill/generative_infill.py#L321"> set the value of `c`</a> to `execute` instead of querying the model.

#### Motion Representation ####

If you'd like to use your own SMPL source motion instead of the ones provided in our `asset_library/`, you'll need to make sure the motion is converted into our data representation. Our data representation has shape (Batch size, 236, 1, 60). The feature dimension is 236, and we handle 60 frame motions.

The 236-dimensional feature vector comprises the following: 24 SMPL joint rotations converted to rotation6d representation and flattened (24*6 = 144), 3D world space translation of the root followed by a 3 zero paddings (3 + 3 = 6). Then we include a 4-dim foot contact label (4) and a 10-dim SMPL body shape parameter (which we just set to 0). Finally, we include the 3D world-space positions of the joints (24 * 3 = 72) as a redundant representation.

Overall, 24*6 + 3 + 3 + 4 + 10 + 24 * 3 = 236.

Check out the <a href="https://github.com/purvigoel/iterative-editing-release/blob/new-branch4/generative_infill/reloader.py#L26">reloader.py</a> for some code on calculating the world space positions and foot contact labels, once you've got your joint rotations.

### Development ###

This is a research prototype and, in general, will not be regularly maintained long after release. 

### Acknowlegements ###
This work was supported by a Stanford Interdisciplinary Graduate Fellowship, the Stanford Wu Tsai Human Performance Alliance, Meta and Activision.
