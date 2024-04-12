# Using docker in this project

## Docker commands

Remove all containers

```shell
docker rm -f $(docker ps -aq)
```

Remove image

```shell
docker rmi <image_name>
```

## Step 1: Install docker

### Windows - Docker

Install docker desktop

### Arch - Docker

```shell
sudo pacman -S docker docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
```

## Step 2: Install nvidia containers

### Windows - Nvidia containers

Just make sure docker desktop uses wsl2 and cuda is up to date

### Arch - Nvidia containers

 ``` shell
yay -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
 ```

## Step 2.5: Change command for sweeps, etc

```Dockerfile
CMD ["Custom command"]
```

## Step 3: Build docker container

```shell
docker build -t <PROJECT_NAME> -f Dockerfile.wandb .
```

## Step 3.5: Create volumes

Create volume called data

```shell
docker run  -v data:/usr/src/app <PROJECT_NAME>
docker run -it --rm -v data:/data alpine sh
mv data/raw .
```

## Step 4: Run docker container

### Linux

```shell
docker run --gpus <all | '"device=0,1"'> -v <project-path>/data:/app/data <PROJECT_NAME>
docker run -it --rm --gpus all -v ~/competitions/q3-harmful-brain-activity/data:/app/data <PROJECT_NAME>:latest
```

### Windows

```shell
docker run --ipc=host -it --gpus all  -v data:/usr/src/app/data <PROJECT_NAME>
```
