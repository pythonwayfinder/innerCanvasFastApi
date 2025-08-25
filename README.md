# wayFinderFastApi

[ 사용법 ]

1. 프로젝트를 넣어둘 새 폴더를 만들어서 visual studio code로 해당 경로 열기

2. 터미널에 python -m venv .venv 입력

3. .\.venv\Scripts\activate 을 터미널에 입력해서 가상환경 진입

3-2. 만약 3 과정에서 권한 혹은 보안 오류가 발생할 경우, 터미널에
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 을 입력한 다음 다시 .\.venv\Scripts\activate 입력

3-3. 터미널에 (.venv)가 보일 경우 가상환경 진입에 성공

4. 가상환경 터미널에 pip install jupyter notebook ipykernel 입력해서 주피터 설치

5. python -m ipykernel install --user --name=venv --display-name "Python (venv)" 입력

6. testcode.ipynb를 마지막 부분을 실행시켜서 c:\finalproject\.venv\Scripts\python.exe이 뜰 경우 성공

기타 : 패키지를 다시 설치해야 한다면 터미널에 pip install -r requirements_package.txt 입력


[ Usage ]

1. Create a new folder for the project and open it with Visual Studio Code.

2. In the terminal, run:
   ```bash
   python -m venv .venv

3. Activate the virtual environment:
   .\.venv\Scripts\activate

3-2. If you encounter a permission or security error in step 3, run:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

    Then try activating again:
    .\.venv\Scripts\activate

3-3. If you see (.venv) in the terminal prompt, the virtual environment has been successfully activated.

4. Inside the virtual environment, install Jupyter:
   pip install jupyter notebook ipykernel

5. Register the virtual environment as a Jupyter kernel:
   python -m ipykernel install --user --name=venv --display-name "Python (venv)"

6. Open and run the last cell of testcode.ipynb.
   If you see c:\finalproject\.venv\Scripts\python.exe in the output, the setup is successful.

Note: If you ever need to reinstall the required packages, run:
      pip install -r requirements_package.txt

