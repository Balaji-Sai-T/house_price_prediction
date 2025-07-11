import plotly.express as px
import plotly.io as pio
import pandas as pd

# Set renderer (optional for saving)
pio.renderers.default = "browser"

# Sample data
df = px.data.iris()

# Simple Plot
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

# Save as PNG
fig.write_image("images/test_output.png")  # Make sure 'images/' exists
fig.show()
