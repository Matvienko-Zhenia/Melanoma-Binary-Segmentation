docker kill segmenaton_service
docker rm segmenaton_service
docker image rm segmenaton_service:v0.1.0
docker build -f Dockerfile -t segmenaton_service:v0.1.0 .
docker run -p 8000:8000 --env-file ./.env --name=segmenaton_service segmenaton_service:v0.1.0