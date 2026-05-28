Implementing some dense-monocular visual odometry methods using python.
Very slow for now, using it mainly to study the algorithms.

## Development container

Build the development image:

```sh
docker build -t python-visual-odometry-dev .
```

Start an interactive development shell with the repository mounted:

```sh
docker compose run --rm dev
```

Run the test suite in the container:

```sh
docker run --rm python-visual-odometry-dev pytest -q
```
