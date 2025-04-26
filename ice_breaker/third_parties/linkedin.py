import os
import requests
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url:str, mock: bool = False):
    """"Scrape LinkedIn profile information from a given URL.
    Args:
        linkedin_profile_url (str): The URL of the LinkedIn profile to scrape.
        mock (bool): If True, return mock data instead of scraping. Defaults to False.
    Returns:
        dict: A dictionary containing the scraped profile information.
    """
    