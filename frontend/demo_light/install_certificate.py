import os
import ssl
import certifi


def install_cert():
    # Get the path to the certifi certificate bundle
    certifi_path = certifi.where()

    # Set the SSL certificate file path environment variable
    os.environ["SSL_CERT_FILE"] = certifi_path
    os.environ["REQUESTS_CA_BUNDLE"] = certifi_path

    # Configure SSL context to use the certifi certificates
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=certifi_path)

    return certifi_path


if __name__ == "__main__":
    cert_path = install_cert()
    print(f"Certificates installed at: {cert_path}")
