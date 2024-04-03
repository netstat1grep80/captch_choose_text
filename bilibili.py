# -*- coding: utf-8 -*-

import asyncio
from playwright.async_api import async_playwright
import os
import sys
import requests

from loguru import logger

__ROOT__ = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(__ROOT__ + os.sep + "libs" + os.sep)
from libs import gol

captch_api = 'http://10.200.148.220:5000/detect_image'
captch_image_path = './tmp/element_screenshot.png'

gol.__init()
gol.set_value('logger', logger)
gol.set_value('__ROOT__', __ROOT__)
gol.set_value('sep', os.sep)
gol.set_value('log', logger)
gol.set_value('captch_img_url', '')
gol.set_value('captch_img_coord', [])
gol.set_value('login_success', False)

async def on_response(response, page):
    if "captcha_v3/batch" in response.url and response.url != gol.get_value('captch_img_url'):
        gol.set_value('captch_img_url', response.url)
        logger.debug("验证码图片：{}".format(response.url))

        # 等待15秒，直到指定的选择器出现
        try:

            await page.wait_for_timeout(2000)
            await page.screenshot(path='./tmp/body.png')
            element = await page.query_selector('body > div.geetest_panel.geetest_wind > div.geetest_panel_box.geetest_panelshowclick')
            await page.wait_for_timeout(2000)
            await element.screenshot(path=captch_image_path)
            await page.wait_for_timeout(1000)
            await detect_capth_image(page, element)
        except Exception as e:
            logger.error(e)

    if "api.geetest.com/ajax.php" in response.url:
        response_text = await response.text()
        logger.debug(response_text)
        if "result" in response_text and "success" in response_text and "validate" in response_text:
            gol.set_value('login_success', True)


async def detect_capth_image(page, element):
    with open(captch_image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(captch_api, files=files)
        ret = response.json()
        logger.debug(ret)
        if ret['code'] == 0:
            is_duplicate = len(ret['coord']) != len(set(tuple(x) for x in ret['coord']))
            if is_duplicate:
                btn_refresh = await page.query_selector('body > div.geetest_panel.geetest_wind > div.geetest_panel_box.geetest_panelshowclick > div.geetest_panel_next > div > div > div.geetest_panel > div > a.geetest_refresh')
                await btn_refresh.click()
            else:
                submit = await page.query_selector('body > div.geetest_panel.geetest_wind > div.geetest_panel_box.geetest_panelshowclick > div.geetest_panel_next > div > div > div.geetest_panel > a')
                bounding_box = await element.bounding_box()
                i = 0
                try:
                    for point in ret['coord']:
                        i = i + 1
                        x = bounding_box['x'] + int(point[0]) + 30
                        y = bounding_box['y'] + int(point[1]) + 30
                        logger.debug("move {} => ({},{})".format(i, x, y))
                        await page.mouse.move(x, y)
                        await page.wait_for_timeout(1000)
                        await page.mouse.down()
                        await page.wait_for_timeout(300)
                        await page.mouse.up()
                        await page.wait_for_timeout(1000)
                        logger.debug("point {} click".format(i))
                    logger.debug("start submit")
                    await submit.click()
                except Exception:
                    btn_refresh = await page.query_selector('body > div.geetest_panel.geetest_wind > div.geetest_panel_box.geetest_panelshowclick > div.geetest_panel_next > div > div > div.geetest_panel > div > a.geetest_refresh')
                    await btn_refresh.click()

        else:
            btn_refresh = await page.query_selector('body > div.geetest_panel.geetest_wind > div.geetest_panel_box.geetest_panelshowclick > div.geetest_panel_next > div > div > div.geetest_panel > div > a.geetest_refresh')
            await btn_refresh.click()


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")  # 关闭爬虫标识
        page = await context.new_page()

        # page.locator(".geetest_item_img").click()
        # page.locator(".geetest_item_img").click()
        # page.get_by_role("link", name="确认").click()
        try:
            page.on('response', lambda response: on_response(response, page))
            await page.goto("https://www.bilibili.com/", wait_until='networkidle')
            await page.get_by_text("登录", exact=True).click()
            await page.wait_for_timeout(1000)
            await page.get_by_placeholder("请输入账号").click()
            await page.wait_for_timeout(1000)

            for char in 'input_your_bilibili_username':
                await page.get_by_placeholder("请输入账号").type(char)
                await page.wait_for_timeout(200)

            await page.get_by_placeholder("请输入密码").click()
            await page.wait_for_timeout(1000)

            for char in 'F3*Yd_4Y8fg4':
                await page.get_by_placeholder("请输入密码").type(char)
                await page.wait_for_timeout(200)

            await page.get_by_text("登录", exact=True).nth(1).click()

            login_success = gol.get_value('login_success')
            while not login_success :
                login_success = gol.get_value('login_success')
                # logger.debug("login status:{}".format(login_success))
                await page.wait_for_timeout(1000)
            logger.success("See u !")

        except Exception as e:
            logger.critical(e)


        await page.wait_for_timeout(30*1000)
        await context.close()
        await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Ctrl+C pressed, exiting gracefully...')
        exit()