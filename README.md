# Data Science Template

This sample repo contains the recommended structure for a Python data science project. For more information on data science in VS Code, see the [Data Science Overview](https://code.visualstudio.com/docs/datascience/overview) in our docs. In this sample, we use the `pandas` and `matplotlib` libraries to perform data analysis and visualize sample data and the `pytest` library to perform tests.

For a more in-depth tutorial, see our [data science tutorial](https://code.visualstudio.com/docs/datascience/data-science-tutorial).

The code in this repo aims to follow Python style guidelines as outlined in [PEP 8](https://peps.python.org/pep-0008/).

## Running the Sample

To successfully run this example, we recommend the following VS Code extensions:
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) 

- Open the template folder in VS Code (**File** > **Open Folder...**)
- Create a Python virtual environment using the **Python: Create Environment** command found in the Command Palette (**View > Command Palette**). Ensure you install dependencies found in the `requirements.txt` file
- Ensure your newly created environment is selected using the **Python: Select Interpreter** command found in the Command Palette
- Run `calculations.py` using the Play Button in the top right corner or by selecting **Python > Python File in Terminal** from the context menu or Command Palette
- Run `revenue_visual.py` using the Play Button in the top right corner or by selecting **Python > Python File in Terminal** from the context menu or Command Palette to generate the bar graph visual
- To test the Python code, install `dev-requirements.txt` into your virtual environment. 
- Navigate to the Test Panel to configure your Python test or by triggering the **Python: Configure Tests** command from the Command Palette
- Run tests in the Test Panel or by clicking the Play Button next to the individual tests in the `test_calculations.py` file

