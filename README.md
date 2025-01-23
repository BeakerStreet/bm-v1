# Buddy Model v1, BM-v1

## Setup Instructions

### Prerequisites

- Python 3

### Virtual Environment Setup

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv myenv
   ```

2. **Activate the Virtual Environment**:
   ```bash
   source myenv/bin/activate
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables (primarily for Google Drive and OpenAI in training)

- Create a `.env` file in the root directory.
- Add environment variables in the following format:
  ```plaintext
  OPENAI_API_KEY=your_key
  GOOGLE_APPLICATION_CREDENTIALS=your_credentials
  ```

### Deactivate the Virtual Environment

- To deactivate the virtual environment, simply run:
  ```bash
  deactivate
  ```

To run the `bm-v1.py` program, follow these steps:

1. **Navigate to the directory** where the `bm-v1.py` file is located.

2. **Execute the program** using Python. 

3. **Follow any on-screen instructions** or prompts provided by the program. You will be required to enter an Image URL that the model can use to generate its prediction.
