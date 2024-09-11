# Steps To run the API

1. Clone this repository to your local machine using the following command:

   ```
   https://github.com/SHAKTHI-VEL/ragFlask.git
   ```

2. Change your current directory to the project folder:

   ```
   cd ragFlask
   ```

3. Run the following command to fetch the dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create the .env file and copy the contents of .env.example into it by typing the following command:
    ```
    cp .env.example .env
    ```

5. Finally start the server by typing the following command:
    ```
    flask --app rag run
    ```