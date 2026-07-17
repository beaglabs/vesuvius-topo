# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#   "marimo>=0.14",
#   "vesuvius-ssm",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Vesuvius SSM unwrapping

        Configure a bounded PHerc0332 experiment. Run the generated commands in a Molab terminal so long GPU jobs and artifacts survive notebook-cell refreshes.
        """
    )
    return


@app.cell
def _(mo):
    bounds = mo.ui.text(value="500,756,300,556,300,556", label="Level-2 bounds (z0,z1,y0,y1,x0,x1)")
    seed = mo.ui.text(value="1600,1600,2400", label="Seed xyz")
    direction = mo.ui.text(value="1,0,0", label="U direction xyz")
    size = mo.ui.slider(64, 512, value=256, step=32, label="Output size")
    mo.vstack([bounds, seed, direction, size])
    return bounds, direction, seed, size


@app.cell
def _(bounds, direction, mo, seed, size):
    prepare_command = f"""vesuvius-ssm prepare --level 2 --bounds {bounds.value} --count 2000 --output artifacts/trajectories"""
    train_command = "vesuvius-ssm train --trajectories artifacts/trajectories --epochs 20 --output artifacts/tracker.pt"
    rollout_command = f"""vesuvius-ssm rollout --level 2 --bounds {bounds.value} --model artifacts/tracker.pt --seed-xyz {seed.value} --u-direction {direction.value} --height {size.value} --width {size.value} --output artifacts/surface.npz"""
    mo.md(
        f"""
        **Prepare**
        ```bash
        {prepare_command}
        ```

        **Train**
        ```bash
        {train_command}
        ```

        **Roll out**
        ```bash
        {rollout_command}
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
