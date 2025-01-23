# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [x] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 80

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

232812, s244501, s232811

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- question 3 fill here ---

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We use pip and conda for package managing and virtual enviorments as it was proposed in the course. 
The list of dependencies are specified in requirements.txt and requirements_dev.txt. 
We use pipreqs to automatically create the requirements.txt
To get an exact copy of the enviorment a new team member must: 
1. clone the git repository
git clone https://github.com/JohnBBB42/dtu_mlops_group_80.git
2. navigate to the project directory 
3. create the conda enviorment 
conda create --name my_environment_name --file requirements.txt
4. activate my_environment_name
5. install dev requirements
pip install -r requirements_dev.txt

This process ensures that all required packages are installed and the environment is consistent across all team members.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized our project using the mlops_template from Nicki's repository (https://github.com/SkafteNicki/mlops_template). The structure includes key folders such as configs, dockerfiles, src/energy, and tests, each serving a specific purpose. The configs folder contains files for managing parameters and experiment setups, while the dockerfiles directory holds Docker configurations. The core functionality of the project is in src/energy, which contains scripts for data processing, model evaluation, and training.

Additionally, we have a models folder to store ONNX model files, including optimized versions, and tests for unit testing. We also included an ml_deployment folder for deployment-specific scripts, such as bentofile.yaml for serving models and locust_file.py for performance testing. While we kept the docs folder from the template, we didn’t have time to use it. The outputs and reports directories are used for saving experimental results and documentation.

Overall, we stayed close to the template but added the ml_deployment folder to better support deployment needs, adapting the structure to fit our project’s specific requirements.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We implemented several rules for code quality and formatting using pre-commit hooks. The configuration includes checks for common issues, such as trailing whitespaces (trailing-whitespace), end-of-file newline consistency (end-of-file-fixer), YAML file syntax validation (check-yaml), large file additions (check-added-large-files), JSON validation (check-json), and detecting unresolved merge conflicts (check-merge-conflict).

For linting and formatting, we employed ruff, which is both a linter and formatter, ensuring compliance with Python’s best practices and maintaining consistent code style across the project.

These tools enhance maintainability and readability, especially in larger projects, by catching issues early and enforcing a uniform structure. Typing and documentation, while not explicitly configured in this file, are also essential as they improve developer understanding, reduce onboarding time, and minimize errors. For example, type hints clarify function expectations, while good documentation ensures clarity on how components interact.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 12 tests across key modules: 

1. **Model Testing**: Verifies output shapes, correctness of training, validation, and testing steps, including loss computation and backpropagation.
2. **Data Testing**: Validates preprocessing (creation of `.pt` files), dataset loading, data integrity, and batch generation in the data module.

These tests ensure core components like preprocessing and model training function as expected, preventing errors from propagating and causing delays in larger projects.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our code is 90%, which includes the most relevant source files to the project. This indicates that a majority of our code has been executed during testing, but 12 lines remain untested. While achieving 100% code coverage would be ideal, even if we reached that level, it would not guarantee the code is error-free. Code coverage only measures how much of the code is executed, not the quality or correctness of the tests themselves. Bugs can still arise from edge cases, incorrect assumptions, or unexpected integrations that the tests fail to address. Therefore, in addition to improving coverage, it’s essential to focus on writing robust test cases that validate functionality, handle edge cases, and account for real-world scenarios to ensure reliability and minimize errors.


### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Yes, we utilized branches and pull requests in our workflow to enhance collaboration and maintain code quality. Direct pushes to the main branch were disabled, ensuring that all changes required a pull request for integration. This enforced a review process, improving oversight and reducing errors.

We also implemented a CI workflow using GitHub Actions. Each pull request triggered automatic tests on multiple operating systems and Python versions. This ensured that code changes were thoroughly validated before merging into the main branch.

For feature development, we created individual branches for each feature or task. After implementing a feature, we merged the latest changes from the feautre branch into the main branch. This practice allowed us to resolve potential merge conflicts early, keeping the main branch clean and stable. By using this structured approach, we streamlined collaboration, tracked individual contributions, and maintained high-quality code throughout the project lifecycle.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC for data version control in our project. It was particularly convenient for ensuring that all team members were working with the same version of the dataset, eliminating discrepancies and streamlining collaboration. By tracking our data with DVC, we could efficiently share large files via remote storage without overloading our Git repository, while still maintaining version history. This also ensured reproducibility, as previous versions of the data could be easily restored if needed.

However, since our dataset was static and not expected to change during the project, we did not require a highly dynamic version control system for the data. For this reason, we did not track the model itself with DVC, focusing only on the dataset. If we had been working with a frequently updated or evolving dataset, DVC would have been even more beneficial for managing changes and testing how different versions of the data impacted model performance. In such cases, it would also be useful for auditing and debugging pipelines by associating model outputs with specific data versions.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have organized our continuous integration (CI) pipeline into three separate workflows: **Code Formatting**, **Pre-commit Checks**, and **Unit Tests**. These workflows are implemented in GitHub Actions to ensure code quality, adherence to coding standards, and functionality.

1. **Code Formatting**:
   - The `codecheck.yaml` workflow ensures code quality by running tools like `Ruff` for linting and formatting and `Mypy` for type checking. It is triggered on every push or pull request to the `main` branch and runs on multiple operating systems (Ubuntu, Windows, macOS) and Python versions (3.11 and 3.12). This workflow also uses pip caching to reduce runtime.

2. **Pre-commit Checks**:
   - The `pre_commit.yaml` workflow ensures that commits adhere to coding standards enforced by pre-commit hooks. It fetches the latest changes, runs the hooks, and automatically commits fixes if necessary. Like the Code Formatting workflow, it tests on multiple operating systems and Python versions to ensure compatibility.

3. **Unit Testing**:
   - The `tests.yaml` workflow runs our unit tests using `pytest` with coverage tracking. It validates the functionality of the data pipeline and model, as seen in `test_data.py` and `test_model.py`. This workflow is also configured to run on Ubuntu, Windows, and macOS with Python 3.11 and 3.12. It leverages dependency caching and provides detailed coverage reports.

By separating these workflows, we ensure modular and efficient CI processes. Testing on multiple OS platforms and Python versions guarantees compatibility across diverse environments. Pip caching further optimizes the runtime of all workflows. This setup provides confidence that our code is high quality and maintains functionality during development. An example workflow file can be found [here](https://github.com/<your-repository>/actions/workflows/tests.yaml).

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured our experiments using a combination of Hydra configuration files and command-line arguments with `Typer`. This allows us to set default hyperparameters in a `config.yaml` file while also enabling overrides via CLI options. For example, we can run an experiment with:

```bash
python train.py --lr 0.001 --batch_size 32 --epochs 50
```

The configuration combines CLI arguments with values from `config.yaml` to ensure flexibility and reproducibility. Additionally, we use tools like PyTorch Lightning for training and WandB for logging, which integrates seamlessly with our setup.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured reproducibility of experiments by using a combination of configuration files, tracking tools, and structured workflows. Experiments were managed with a `config.yaml` file that stores hyperparameters and settings, which can be reused to recreate runs. We tracked every experiment with Weights & Biases (WandB), logging configurations, results, and metadata, such as the code version, system environment, and dataset used (e.g., details in `wandb-metadata.json` and `wandb-summary.json`).

The exact Python environment and dependencies were recorded in `requirements.txt`, ensuring that experiments can be reproduced in identical environments. To replicate any experiment, one simply needs to load the logged configuration in WandB or run the `train.py` script with the stored config file.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

In our experiments, we tracked critical metrics and hyperparameters using WandB. The first screenshot shows the **hyperparameter_tuning** for different hyperparameter sweeps, such as learning rate (`lr`), batch size, and epochs. Validation loss is crucial as it measures model performance on unseen data, guiding us in selecting the best hyperparameters. Notably, the plot reveals signs of overfitting in certain sweeps where validation loss increases sharply after an initial stable phase. This highlights the importance of early stopping and regularization in our training process to prevent overfitting.

[validation_loss](figures/validation_loss.png)


The second screenshot illustrates the **importance and correlation of hyperparameters** with validation loss. Batch size emerged as the most influential parameter, showing a strong negative correlation with loss, meaning smaller batch sizes likely resulted in better validation performance. Such insights help refine our experiments by focusing on impactful hyperparameters while deprioritizing less critical ones. This analysis was essential in efficiently navigating the parameter space.


[hyperparameter_importance](figures/hyperparameter_importance.png)


The third screenshot depicts a **parallel coordinate plot** of hyperparameter combinations and their corresponding validation loss. This visualization reveals how specific configurations (e.g., lower batch sizes paired with moderate learning rates) lead to improved performance, while certain combinations show suboptimal results. These observations enabled better decision-making during hyperparameter optimization by clearly visualizing the trade-offs and relationships between parameters.

[hyperparameter_tuning](figures/hyperparameter_tuning.png)


These metrics and visualizations were vital in understanding the effects of different configurations on model performance, ensuring we identified the best-performing setup. Tracking these allowed us to iteratively improve the model, address overfitting issues, and maintain experiment reproducibility with a clear audit trail of parameter choices and their outcomes.


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging in our project primarily still relied on a combination of **print statements**, **GenAI tools**, and the course material. Print statements were quick and effective for identifying issues, especially in smaller functions or during initial testing phases. GenAI tools, such as ChatGPT, provided additional insights and suggestions, helping us resolve more complex issues efficiently. The course material and slack also served as a reliable reference for debugging common errors related to the frameworks and tools we used.

We performed a single profiling run using PyTorch’s built-in profiler to evaluate the efficiency of key components, such as the data pipeline and model training loops. This profiling helped identify a few bottlenecks, such as unnecessary data loading redundancies, which were then optimized. While the code is now functional and performs well, we believe further profiling could reveal additional opportunities for fine-tuning performance. Debugging and profiling were critical in ensuring our experiments ran smoothly and efficiently.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:


The biggest challenge in the project was the sheer number of new software tools we had to learn and use, which often felt overwhelming. At times, multiple solutions were proposed for the same issue, making it difficult to decide on the best approach.

Despite these struggles, the course provided an invaluable opportunity to learn and adapt. To overcome challenges, we relied on the course material, collaborated closely as a group, and used GenAI tools like ChatGPT for guidance. Breaking tasks into smaller, manageable pieces and focusing on practical solutions helped us push through moments of confusion.

To build on this experience, it would be beneficial to integrate these concepts into Business Analytics classes across the program. For instance, rather than relying heavily on Jupyter notebooks in analytics courses, the setup provided by this course, such as Nicki’s MLOps template, could be used as the foundation.

A potential integration could involve splitting the focus across multiple classes. The first class could focus on coding structure, reproducibility, and version control, introducing students to tools like Git, DVC, and configuration management using Hydra. Students would learn how to structure projects for long-term scalability, manage data versioning, and ensure experiments can be reproduced.

The second class could emphasize experiment tracking and optimization with tools like WandB. Students would learn to log metrics, perform hyperparameter sweeps, and analyze results effectively, which are crucial for iteratively improving machine learning models.

Finally, a third class could delve into cloud deployment and applications, focusing on tools like Docker, Kubernetes, and cloud platforms (e.g., AWS, GCP, or Azure). This would teach students how to deploy models in production environments and scale machine learning workflows effectively.

By integrating these components into the Business Analytics curriculum, students would gain a gradual and well-rounded understanding of MLOps, reducing the steep learning curve while reinforcing these critical concepts through hands-on practice.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
