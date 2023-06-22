import plotly.express as px
import pandas as pd

def plot_molecule(pos, charges):
    """
    Return a 3D plot of a molecule
    pos: ndarray of X,Y,Z atoms' position 
    charges: ndarray of atoms' charges
    """
    pos = [p for p in pos if any(p != 0)]
    charges = [c for c in charges if c != 0]
    df = pd.DataFrame(pos, columns=["x", "y", "z"])
    df = df.join(pd.Series(charges, name="a"))
    return px.scatter_3d(df, "x", "y", "z", "a")