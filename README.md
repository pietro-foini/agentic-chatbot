# Agentic-Chatbot

Hello there! 😀

This project is an **Agentic Chatbot** built using **LangChain**, **LangGraph**, and **FastAPI**, with **PostgreSQL**
(enhanced by the `pgvector` extension) as the core data layer for semantic vector search, long-term memory, and
conversation state persistence.

The application is structured to be cloud-agnostic and modular: it can run fully in a local development environment, or
be deployed to **Google Cloud Platform (GCP)** and **Amazon Web Services (AWS)**.

The core system coordinates agents through state graphs (LangGraph), allowing:

- Dynamic tool-calling decisions.
- Multi-modal capabilities.
- Vector retrieval and similarity ranking stored natively inside PostgreSQL with `pgvector`.
- High-throughput asynchronous endpoints powered by FastAPI.

The purpose of the application is mainly to provide a prototype and a tutorial for implementing a simple web
application.

## Installation

This project requires **Python 3.14** and relies strictly on [Poetry](https://python-poetry.org/) for modern packaging
and deterministic dependency resolution.

### Application

In order to isolate the dependencies based on the desired setup, we offer different ways of configuring the project
dependencies.

#### Local

Installs the core libraries (e.g. FastAPI, LangChain, LangGraph, etc.) alongside the LangChain Ollama implementation for
running models locally. This setup is useful for a fully local environment, with no need to connect to any cloud
service, not even for API calls to LLM or embeddings models. To install, run the following command:

```bash
poetry install --extras "local"
```

As a side note, depending on the chosen model, this setup requires a local machine with good computing power,
preferably with a GPU. For example, with an NVIDIA RTX 5060 8GB, the `qwen2.5:7b` model works great, while the
`gemma4:12b` model already struggles a bit.

To get these models, first follow the Ollama documentation to download them locally: https://ollama.com/download.

#### GCP

Installs the core libraries alongside Google Cloud client packages. This setup is certainly useful for testing a
possible future deployment on GCP, but it is also useful for simply making API calls to Google's Gemini models. As we
will see later, we can use this feature while still relying on a fully local database service. To install, run the
following command:

```bash
poetry install --extras "gcp"
```

To use this setup, you need to have created a GCP project with an active billing account (or within the free trial
period).

#### AWS

Installs the core libraries alongside AWS SDKs. This setup is certainly useful for testing a possible future
deployment on AWS, but it is also useful for simply making API calls to Amazon's Nova models. As we will see later, we
can use this feature while still relying on a fully local database service. To install, run the following command:

```bash
poetry install --extras "aws"
```

To use this setup, you need to have created an AWS account with an active billing account (or within the free trial
period).

#### All

Installs every module across local, GCP, and AWS environments. Useful for monolithic test environments or central
workstations. To install, run the following command:

```bash
poetry install --extras "all"
```

### Database

To run database capabilities locally, make sure to have a running PostgreSQL instance
(https://www.postgresql.org/download/), which will have you create a default `postgres` user with its password.
This local instance is obviously needed for the local setup, but also for connecting to cloud databases, such as Cloud
SQL on GCP and RDS on AWS.
Indeed, once these cloud resources are created, they can also be used, tested, and monitored locally through suitable
connections, which we will see later for each provider.

## Local Deployment

Using any graphical client (e.g. DBeaver) for the local configuration, we create a connection to PostgreSQL on port
5432 and create a database:

```sql
CREATE DATABASE chatbot;
```

Connect to the database and enable the `pgvector` vector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

This extension is mandatory: it adds native vector similarity data types and distance operators (`<->`, `<#>`, `<=>`).

To run the code locally once the desired installation steps are completed, you need to configure the environment
variables in a dedicated `.env` file, depending on the chosen setup. The variables to set are the following:

```
CLOUD_PROVIDER
STORAGE_BUCKET
SECRET_KEY
COOKIE_SECURE
LLM_MODEL
EMBEDDINGS_MODEL
DB_USER
DB_PASSWORD
DB_NAME
DB_HOST
DB_PORT
```

Suppose we want to test a fully local configuration. In this case, one way of populating the variables could be the 
following:

```
CLOUD_PROVIDER=none
STORAGE_BUCKET=your_bucket_name
SECRET_KEY=your_super_secret_jwt_or_app_key
COOKIE_SECURE=False
LLM_MODEL=qwen2.5:7b
EMBEDDINGS_MODEL=bge-m3:latest
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_NAME=chatbot
DB_HOST=localhost
DB_PORT=5432
```

Suppose we want to test the configuration on GCP while relying on a fully local database. On GCP, all we have done is
create the project and a simple bucket called `your_bucket_name` from the GCP Console. In this case, one way of
populating the variables could be the following:

```
CLOUD_PROVIDER=gcp
STORAGE_BUCKET=your_bucket_name
SECRET_KEY=your_super_secret_jwt_or_app_key
COOKIE_SECURE=False
LLM_MODEL=gemini-2.5-flash-lite
EMBEDDINGS_MODEL=models/gemini-embedding-001
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_NAME=chatbot
DB_HOST=localhost
DB_PORT=5432
```

If we wanted to do the equivalent with AWS, it is enough to create a bucket on S3:

```
STORAGE_BUCKET=your_bucket_name
SECRET_KEY=your_super_secret_jwt_or_app_key
COOKIE_SECURE=False
LLM_MODEL=amazon.nova-lite-v1:0
EMBEDDINGS_MODEL=amazon.titan-embed-text-v2:0
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_NAME=chatbot
DB_HOST=localhost
DB_PORT=5432
```

We can then run the application locally. Activate the virtual environment and run the Uvicorn ASGI server with
live reload enabled:

```bash
uvicorn chatbot.main:app --port 8000 --env-file .env --reload
```

The application will listen on http://localhost:8000.

## Docker Deployment

To dockerize the application into a Docker image, we use Docker Compose to manage the database and the application
together. Dockerization is supported only for the GCP setup:

- Create a `.secrets` folder.
- Copy the `application_default_credentials.json` file you have in
  `%APPDATA%\gcloud\application_default_credentials.json` (or the JSON key of your Service Account) into the newly
  created folder.

Use the following command to build and start the containers:

**Linux / macOS**

```bash
ENV_FILE=.env docker compose up --build
```

**Windows (PowerShell)**

```powershell
$env:ENV_FILE=".env"; docker compose up --build
```

Use the GCP `.env` configuration and add a `GOOGLE_CLOUD_PROJECT` environment variable pointing to the desired GCP
project whose credentials you have also passed.
This will automatically launch the application, which will be available at http://localhost:8000.

> [!WARNING]
> Mounting your personal Application Default Credentials into the container is fine for local development, but it
> must **not** be done in production. There, rely on a dedicated Service Account attached to the runtime (e.g. Cloud
> Run). Also make sure `.secrets/` and `.env` are listed in your `.gitignore`, so that credentials are never committed.

## Google Cloud Platform (GCP) Deployment

An active Google Cloud Platform project with billing enabled and sufficient IAM permissions (e.g., `Owner` or `Editor`).

Throughout this guide, `<PROJECT_ID>` refers to your GCP project and `<REGION>` to its main region (e.g., `europe-west1`).

> [!NOTE]
> Before launching resources on Google Cloud, you can estimate the monthly operating costs using the official
> [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator?hl=en).

> [!IMPORTANT]
> The entire cloud infrastructure described below has been defined and automated using **Terraform**. The complete
> Infrastructure as Code (IaC) configuration is available in the following repository:
> [pietro-foini/terraform-gcp](https://github.com/pietro-foini/terraform-gcp).

While Terraform provisions the resources deterministically, the sections below explain how each service is structured,
how the pieces fit together, and how you can manage or inspect them manually.

### Cloud SQL Setup

On GCP, the relational database layer is managed by Cloud SQL for PostgreSQL.

- Navigate to Cloud SQL in the GCP Console.
- Click Create Instance, choose PostgreSQL, and specify an Instance ID (e.g., `postgres`), an admin password for the
  default `postgres` user, and your target region. Resource Tier: choose the machine size that fits your budget. 
  For testing, the smallest available tier is enough.
- Open Cloud SQL Studio in the browser, log in with the `postgres` user, and create the production database:

  ```sql
  CREATE DATABASE chatbot;
  ```

- Connect to the `chatbot` database and activate the vector extension:

  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

Now that we have created the resource, we can use it locally, for example, through the **Cloud SQL Auth Proxy**.

#### How does it work?

The proxy runs locally and simulates a local PostgreSQL database on your machine. It listens on a local port (e.g.,
`127.0.0.1:5434`).
When your local application or database UI sends packets to `127.0.0.1:5434`, the proxy encrypts the traffic using TLS
with your GCP IAM credentials and routes it securely through Google's internal APIs directly to your Cloud SQL
instance. Your local Python code does not need to know where the database physically lives: it simply connects to
`localhost`.

#### Setup Steps

- Download the `cloud-sql-proxy` executable for your operating system
  (https://docs.cloud.google.com/sql/docs/mysql/sql-proxy). Place the binary into a persistent folder on your machine
  and add that folder to your system `PATH`.
- Start the proxy from your terminal:

  ```bash
  cloud-sql-proxy <PROJECT_ID>:<REGION>:postgres --port 5434
  ```

  (Keep this terminal window running as long as you need the connection active.)

You can now connect your local GUI (DBeaver) or run migration scripts by targeting host `127.0.0.1`, port `5434`, user
`postgres`, and your set password.

For this configuration, i.e. running locally with a connection to Cloud SQL, it is enough to set these environment
variables:

```
CLOUD_PROVIDER=gcp
STORAGE_BUCKET=your_bucket_name
SECRET_KEY=your_super_secret_jwt_or_app_key
COOKIE_SECURE=False
LLM_MODEL=gemini-2.5-flash-lite
EMBEDDINGS_MODEL=models/gemini-embedding-001
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_NAME=chatbot
DB_HOST=127.0.0.1
DB_PORT=5434
```

### Cloud Run

Create a serverless service that runs the container on which our actual application live.

- Navigate to Cloud Run Services in the GCP Console.
- Create service:
  - Deploy one revision from an existing container image.
  - Select a demo image: `hello`.
  - Define the name `chatbot-service` and the region `<REGION>`.
  - Under Containers, Networking, Security, select Cloud SQL connections and specify the connection (e.g.,
    `<PROJECT_ID>:<REGION>:postgres`). Cloud Run automatically mounts a Unix socket at
    `/cloudsql/<INSTANCE_CONNECTION_NAME>`, giving your container secure, high-speed access to PostgreSQL without public
    routing.

### Artifact Registry

On GCP, Docker images are stored in the Artifact Registry resource.

- Navigate to Artifact Registry in the GCP Console.
- Create a Docker repository (e.g., `ml-servers`) inside your chosen region (e.g., `<REGION>`) to host the
  production container images built for Cloud Run.

### Secret Manager

On GCP, secret variables that should not be exposed are managed by the Secret Manager service.

- Navigate to Secret Manager in the GCP Console.
- Create three separate secret keys in GCP Secret Manager:
  - `postgres_db_password`
  - `postgres_db_username`
  - `secret_key`
- Populate them.

These variables will be used and configured to the Cloud Run during deployment (see next step and the file `cloud/gcp/cloudbuild.yaml`).

### Cloud Build

The deployment pipeline is fully automated via Cloud Build:

- Connect your GitHub repository to GCP using the Developer Connect API.
- Navigate to IAM & Admin and verify that the Cloud Build Service Account (or dedicated worker account) possesses the
  necessary IAM roles:
  - Agent Platform User
  - Artifact Registry Writer
  - Cloud Build Service Account
  - Cloud Run Admin
  - Cloud SQL Client
  - Developer Connect Read Token Accessor (Beta)
  - Logs Writer
  - Secret Manager Secret Accessor
  - Service Account User
  - Storage Object User
- Create a Cloud Build Trigger:
  - Link the trigger to the connected GitHub repository.
  - Set the event to push on your target branch (e.g., `master`).
  - Define the required Substitution Variables, for example:
    - `_STORAGE_BUCKET`
    - `_COOKIE_SECURE`
    - `_LLM_MODEL`
    - `_EMBEDDINGS_MODEL`
    - `_DB_NAME`
    - `_INSTANCE_CONNECTION_NAME`

The build process reads `cloud/gcp/cloudbuild.yaml`, builds the container, pushes it to Artifact Registry, and deploys
a new revision directly to Cloud Run. The service outputs a secure HTTPS URL where the application is live.

## on Amazon Web Services (AWS) Deployment
