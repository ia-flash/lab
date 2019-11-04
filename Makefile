# The Makefile defines all builds/tests steps

# include .env file
include docker/conf.list

# compose command to merge production file and and dev/tools overrides
COMPOSE?=docker-compose -p $(PROJECT_NAME) -f docker-compose.yml

export COMPOSE
export APP_PORT
export NOTEBOOK_PORT
export NVIDIA_VISIBLE_DEVICES

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

docker/env.list:
	# Copy default config
	cp docker/env.list.sample docker/env.list

docker/conf.list:
	# Copy default config
	cp docker/conf.list.sample docker/conf.list

dev: docker/env.list docker/conf.list
	$(COMPOSE) up

up: docker/env.list docker/conf.list
	$(COMPOSE) up -d

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down --remove-orphans

build:
	$(COMPOSE) build  --no-cache

exec:
	$(COMPOSE) exec torch-notebook bash

logs:
	$(COMPOSE) logs -f --tail 50

test:
	$(COMPOSE) exec torch-notebook pytest 

docs/html:
	$(COMPOSE) exec torch-notebook make -C /workspace/iaflash/docs html

docs: docs/html
	echo "Post"
