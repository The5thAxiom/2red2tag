## Getting Started

These instructions will guide you on how to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- Python 3.x
- virtualenv

```bash
pip install virtualenv
```

### Installing

A step by step series of examples that tell you how to get a development environment running.

1. **Clone the repository**

```bash
git clone https://github.com/Hackathon2024-March/redtag
```

2. **Navigate into the project directory**

```bash
cd redtag\code\model
```

3. **Create a virtual environment**

```bash
virtualenv venv
```

4. **Activate the virtual environment**

- On Windows:

```bash
.venv\Scripts\activate
```

- On Unix or MacOS:

```bash
source venv/bin/activate
```

5. **Install the required dependencies**

```bash
pip install -r requirements.txt
```

### Preprocessing the Data

Before training the model, preprocess your data with the following commands:

1. For AI data:

```bash
python preprocess.py --folder_path "path_to_ai_data" --save_path "processed_ai"
```

2. For human data:

```bash
python preprocess.py --folder_path "path_to_human_data" --save_path "processed_human"
```

### Training the Model

Finally, to train your model, run the following command:

```bash
python train.py
```

