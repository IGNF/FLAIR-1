# Installation

## Env variables (optional)
* Copy the .env.template file into a .env file
* It enables you to change default values for constants

## On Docker
* Install / start docker desktop on your system (https://docs.docker.com/desktop/setup/install/mac-install/)

## On Mac
* Create the env with conda : `conda env create -f environment.yml`
* Activate your env : `conda activate flair`
* Install the landcover : `pip install -e ".[dev]"`

# Usage

You can download postman collections and environments to use the api here : [API-FLAIR-1](https://www.notion.so/netcarbon/API-FLAIR-1-170664b6168e8061a23adb46463a2d77?pvs=4)

## With Docker
* Start the api on your local computer : `docker compose up --build` (you can remove --build if you don't want to force image build)

## With Mac
* Start the api on your local computer : `uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload`
