# wsgi.py

# Import the Flask application instance named 'app'
# from your main application file 'final.py'
from final import app

if __name__ == "__main__":
    # This block is typically used for running the development server directly.
    # A WSGI server like Gunicorn will directly use the 'app' object imported above.
    # You generally wouldn't run `python wsgi.py` in production.
    # Set host and port as needed for development testing here, or rely on Flask defaults.
    # Example: app.run(host='0.0.0.0', port=5001)
    app.run()
