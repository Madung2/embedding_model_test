import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    📌 데이터 입력
    """)
    return


@app.cell
def _(mo):
    file_area = mo.ui.file(kind="area")
    return (file_area,)


@app.cell
def _(file_area):
    file_area
    return


if __name__ == "__main__":
    app.run()
