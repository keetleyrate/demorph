## Using demorph

`demorph` (short for dendrite-morphology) is a tool for tracking the morphology of imaged neurons. `demorph` allows users to select specific cells for tracking across a series of frames and quickly configure case-dependent parameters via a GUI. The program executes the algorithm across all frames in parallel over multiple CPUs.

---

### Step up and preprocessing

To process a video of dendrite growth, the video must be saved as individual frames with the directory structure: `video_dir\frame_dir\frame_x.ext`. 

* **`video_dir`**: Contains directories containing a number of frames.
* **`frame_dir`**: A directory containing the individual frames.
* **`x`**: The frame number.
* **`.ext`**: The image format extension (`.png`, `.jpg`, `.tif`, etc.).

If denoising and contrast equalization is needed, we can pre-process images using:

```\texttt{python demorph.py video_dir/frame_dir -pp --n 4}```

This will write preprocessed copies of the frames into `video_dir/frame_dir` using the number of cores specified by the `--n` argument.

---

### Using the GUI

`demorph` provides an interface where the user can select cells to track and tune detection parameters, case by case. Once the setup is exported, it creates a persistent 'job' object that can be run via the command line. To open the GUI with a video, run:

```\texttt{python demorph.py video_dir/frame_dir --color b --n 4}```

* **`--color b`**: Detects dendrites with a dark color compared to the background.
* **`--color w`**: Detects light dendrites.

#### Cell detection area selection
First, we must select which cells we want to be tracked over the course of the acquisition. This is done with the following steps:

1. Click on the cell or cycle through detected cells using the **Last Cell** and **Next Cell** buttons.
2. Use the **Frame** slider to scrub through the acquisition. 
3.  Use the **Radius** slider to adjust the detection extent around the cell. 
4. Press the **Add Cell** button.
5. Repeat 1-5 until all intended cells have been added.
6. Press the **Write Ridges** button to write auxiliary files to a cache required for further processing.

#### Configure dendrite detection parameters
Next, configure the parameters associated with dendrite detection:
* Low and high ridge hysteresis thresholds.
* Temporal window size.
* Temporal fall-off parameter.

**Factors to consider:**
* **Thresholds**: A tighter fit (smaller low threshold) is more accurate but can "cut off" dendrites from each other.
* **High Threshold**: Keep this small; large values can cause entire branches to be disregarded in some frames.
* **Temporal Window**: Larger windows should generally use a smaller $\alpha$ value to avoid "ghost" dendrites from previous frames.

#### Configure skeletonization parameters
Lastly, configure the parameters associated with skeletonization:
* Minimum distance between local ridge maxima.
* Relative threshold for selecting maxima.
* Number and position of initial skeleton points.

**Factors to consider:**
* **Maxima**: More maxima usually result in a better skeleton, but avoid placing them in non-dendrite areas.
* **Initial Points**: Place these in the center of the tracked cell. Since the path search is directed toward these points, paths which lead to the center of the cell, generally result in better skeletons.

Press the **Export Job** button to save these parameters, and write a 'job' object.

---

### Running Jobs

To run the algorithm on an acquisition, execute the associated job object using the `-r` flag:

```\texttt{python demorph.py video_dir/frames_dir -r --n 8}```

The results are saved as serialized [networkx](https://networkx.org/en/) graph objects in the following directory:

`results/video_dir/frames_dir/cellx/networks/frame_y_network.pkl`

The `networkx` graphs can then be loaded with `pickle` for further analysis.
