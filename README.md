https://athletix.run/challenges/czaMEOxQGg

```sh
docker-compose run --rm app bash app train 0
docker-compose run --rm app bash app train 1
docker-compose run --rm app bash app train 2
docker-compose run --rm app bash app train 3

docker-compose run --rm app bash app submit 0 1 2 3
```
