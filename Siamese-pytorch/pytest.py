# -*- coding: utf-8 -*-
import asyncio
import json
import os
import getopt
import sys
import time
from loguru import logger
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
       
        browser = await p.chromium.launch( headless=False,devtools=False)
        # browser = await p.webkit.launch(headless=False)
        context = await browser.new_context()
        # context = await browser.new_context(record_video_dir='./video')
        await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")  # 关闭爬虫标识

        page = await context.new_page()
        
        try:
           await page.goto('https://bilibili.com', wait_until='networkidle')
           
        except Exception as e:
            logger.critical(e)
        
        await page.wait_for_timeout(30000)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
