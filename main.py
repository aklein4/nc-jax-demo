# one process, multiple devices
# ("data", "fsdp") mesh
# fully replicated parameters
# data parallelism across devices

import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)


BATCH_SIZE = 128
WIDTH = 384


def loss_fn(params, inputs):
    w1, w2 = params
    predictions = jax.nn.gelu(inputs @ w1) @ w2
    return jnp.mean((predictions - 1.0) ** 2)


def main():

    # setup
    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))

    if jax.process_count() != 1:
        raise ValueError("run one Python process with all local devices visible")
    if BATCH_SIZE % len(devices):
        raise ValueError("BATCH_SIZE must be divisible by the number of devices")

    mesh = Mesh(np.asarray(devices).reshape(1, -1), ("data", "fsdp"))
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P(("data", "fsdp")))

    logger.info("node=%s", os.environ.get("NODE_NAME", "local"))
    logger.info("jax=%s", jax.__version__)
    logger.info(
        "backend=%s device_types=%s",
        jax.default_backend(),
        sorted({device.device_kind for device in devices}),
    )
    logger.info("devices=%s mesh=%s", devices, mesh)

    # init params
    rng = np.random.default_rng(42)
    params = tuple(
        jax.device_put(
            rng.standard_normal(shape, dtype=np.float32)
            / np.sqrt(np.float32(shape[0])),
            replicated,
        )
        for shape in ((WIDTH, WIDTH), (WIDTH, WIDTH))
    )

    # init optimizer
    optimizer = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.0)
    with jax.set_mesh(mesh):
        opt_state = optimizer.init(params)

    jax.block_until_ready((params, opt_state))
    logger.info("initialization complete")

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    with jax.set_mesh(mesh):
        step = 1
        while True:

            start = time.monotonic()

            # random inputs
            inputs = rng.standard_normal((BATCH_SIZE, WIDTH), dtype=np.float32)
            batch = jax.make_array_from_process_local_data(batch_sharding, inputs)

            # if step == 1:
            #     logger.info("Compiling first training step...")
            #     train_step = train_step.lower(params, opt_state, batch).compile()
            #     logger.info("Compilation complete; executing first training step...")

            params, opt_state, loss = train_step(params, opt_state, batch)
            loss = float(jax.device_get(loss))

            logger.info(
                "step=%d loss=%.6f seconds=%.6f", step, loss, time.monotonic() - start
            )
            step += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
