from multiapp import MultiApp
from apps import upload, detect, repair


# Init applications
app = MultiApp()

# Add applications here
app.add_app("Upload", upload.app)
app.add_app("Detect", detect.app)
app.add_app("Repair", repair.app)

# Run landing application
app.run()
