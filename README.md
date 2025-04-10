# Legal Anomaly Detection

This project uses embedding techniques to detect anomalies in legal documents.

## Project Structure
```
legal-anomaly/
├── .env # Environment variables
├── app.py # Simple OpenAI API benchmark
├── database.py # PostgreSQL database operations
├── embedding_processor_alternative.py # Free alternative using Sentence Transformers
├── main.py # Main application entry point
├── mock_openai.py # Mock implementation for testing without API keys
├── requirements.txt # Python dependencies
└── environment.yml # Conda environment specification
```

## Workflow

1. **Setup**: Initialize database and environment
2. **Simple Benchmark**: Analyze contracts using only OpenAI API
<!-- 3. **Hybrid Approach**: 
   - Store documents and clauses in PostgreSQL
   - Generate embeddings for similarity search
   - Combine similar examples with OpenAI API for enhanced analysis -->

## Environment Setup

### Create Virtual Environment

```
conda create -n legal-anomaly python=3.10
```

### Install Dependencies

```
conda install -c conda-forge psycopg2 python-dotenv numpy openai
conda install -c conda-forge sentence-transformers
```

### Alternative: Install from Files

```
conda env create -f environment.yml
pip install -r requirements.txt
```

### Update Dependency Files

After adding new packages, update the dependency files:

```
conda env export --from-history > environment.yml
pip freeze > requirements.txt
```

## Database Setup

Ensure PostgreSQL is installed and running. To make sure we are all working on the same postgres database
I have created a docker container which you can use to run and test workflow:

You can run 

```
bash start_db.sh or chmod u+x start_db.sh then do ./start_db.sh 
```

Same for stop_db.sh

Once you have that, you should have a docker container running at which point you have the below commands
to see and play around with the container:

```
# connect to the container
docker exec -it legal_anomaly_db psql -U myuser -d legal_anomaly_db
# see tables
\dt
# see extensions
\dx
# connect to container shell
docker exec -it legal_anomaly_db bash
# see running docker processes so that you can verify that the container is running
docker ps
```

## Running the Application

To test that everything has been setup properly you can run the following command:

```
python test_full_workflow.py
```

and then you can also try and run the following to see how you can use main. All the stuff are mocks so after we have API key, we can begin on next parts. But at least this sets up the environment and ensures workflow is setup.

```
python3 main.py --analyze "Check this text"
```



