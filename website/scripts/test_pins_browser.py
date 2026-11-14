"""Local/CI/production pin checks. Block analytics before executing any page JS."""
import argparse,asyncio,json,os,re,time
from pathlib import Path

async def run(args):
    from playwright.async_api import async_playwright
    result={'base_url':args.base_url};errors=[];tracking=[]
    async with async_playwright() as p:
        launch={'headless':True}
        if os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE'):launch['executable_path']=os.environ['PLAYWRIGHT_CHROMIUM_EXECUTABLE']
        browser=await p.chromium.launch(**launch)
        context=await browser.new_context(viewport={'width':1440,'height':1050})
        async def block(route):
            tracking.append(route.request.url);await route.abort()
        await context.route(re.compile(r'https://[^/]*umami\.is/'),block)
        await context.route(re.compile(r'https://fonts\.(googleapis|gstatic)\.com/'),lambda route:route.abort())
        page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        for _ in range(40):
            try:await page.goto(args.base_url+'/playground/?lang=tok&demo=flow'+('&release='+args.release if args.release else ''),wait_until='domcontentloaded');break
            except Exception:await asyncio.sleep(.5)
        else:raise RuntimeError('The preview server did not become reachable.')
        await page.wait_for_selector('.word-pin',timeout=30000)
        if args.expect_analytics:
            tracker=page.locator('script[src="https://cloud.umami.is/script.js"]');assert await tracker.count()==1
            assert await tracker.get_attribute('data-domains')=='wikilangs.org,www.wikilangs.org'
        initial=await page.locator('.path-word').all_text_contents();assert len(initial)>=5
        # Fix both ends. The middle must retain several valid alternatives.
        await page.locator('.word-pin').nth(0).click();await page.locator('.word-pin').nth(4).click()
        assert await page.locator('.pinned-anchor').count()==2
        assert await page.locator('.future-stage').evaluate('(el)=>Math.abs(el.getBoundingClientRect().width-el.querySelector("svg").viewBox.baseVal.width)<1')
        pinned={0:initial[0],4:initial[4]};result['initial']=initial;seen=set()
        for _ in range(8):
            await page.locator('#pin-resample').click()
            words=await page.locator('.path-word').all_text_contents()
            for pos,word in pinned.items():assert words[pos]==word
            seen.add(tuple(words))
        result['distinct_paths']=len(seen);assert len(seen)>1
        # Changing a constrained branch must still preserve pins.
        candidates=page.locator('.future-node[data-chosen="false"]')
        if await candidates.count():
            await candidates.first.click();words=await page.locator('.path-word').all_text_contents()
            for pos,word in pinned.items():assert words[pos]==word
        # Pins survive playback, undo, a new context, and switching experiences.
        await page.locator('#branch-play').click();await page.wait_for_timeout(1200);await page.locator('#branch-play').click()
        for button in ['#branch-reset','#branch-step']:
            await page.locator(button).click()
            for pos,word in pinned.items():assert (await page.locator('.path-word').all_text_contents())[pos]==word
        if not await page.locator('#branch-undo').is_disabled():await page.locator('#branch-undo').click()
        await page.locator('[data-demo="weaver"]').click();assert await page.locator('.pinned-anchor').count()==2
        words=await page.locator('.path-word').all_text_contents()
        for pos,word in pinned.items():assert words[pos]==word
        await page.locator('[data-demo="flow"]').click();assert await page.locator('.pinned-anchor').count()==2
        # An explicit impossible word has a real empty state, not a fake sentence.
        await page.locator('#pin-position').fill('3');await page.locator('#pin-word').fill('NOT_IN_THIS_CORPUS_7b239');await page.locator('#pin-apply').click()
        assert await page.locator('.pin-no-path').is_visible()
        assert await page.locator('.future-node').count()==0
        assert await page.locator('.path-word').count()==0
        assert await page.locator('#pin-resample').is_disabled()
        await page.locator('[data-demo="weaver"]').click();assert await page.locator('.pin-no-path').is_visible()
        assert not (await page.locator('.weaver-passage').inner_text()).strip()
        # Clearing all pins restores a live graph and valid position numbering.
        await page.locator('#pin-clear').click();await page.wait_for_selector('.future-node')
        assert await page.locator('.pinned-anchor').count()==0
        assert max(int(x)for x in await page.locator('.future-node').evaluate_all('(els)=>els.map(e=>e.dataset.position)'))<20
        await page.locator('[data-demo="flow"]').click()
        # An arbitrary future position extends the search while preserving its pin.
        words=await page.locator('.path-word').all_text_contents()
        await page.locator('#pin-position').fill('7');await page.locator('#pin-word').fill('li');await page.locator('#pin-apply').click()
        result['future_pin_connected']=not await page.locator('.pin-no-path').is_visible()
        if result['future_pin_connected']:
            assert len(await page.locator('.path-word').all_text_contents())==7
            assert (await page.locator('.path-word').all_text_contents())[6]=='li'
        await page.locator('#pin-clear').click()
        # RTL and mobile keep the pins usable and avoid page overflow.
        await page.locator('#corpus-language').select_option('ary');await page.wait_for_function('document.querySelector("#playground-content").lang==="ary"');await page.wait_for_selector('.word-pin')
        await page.locator('.word-pin').nth(2).click();await page.locator('.word-pin').nth(4).click()
        await page.set_viewport_size({'width':390,'height':844});await page.wait_for_timeout(600)
        assert not await page.evaluate('document.documentElement.scrollWidth>innerWidth')
        assert await page.locator('.pinned-anchor').count()==2
        if args.report:
            report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True)
            await page.screenshot(path=str(report.with_suffix('.mobile.png')),full_page=True)
        await page.set_viewport_size({'width':1440,'height':1050});await page.locator('#corpus-language').select_option('tok');await page.wait_for_function('document.querySelector("#playground-content").lang==="tok"');await page.wait_for_selector('.word-pin')
        assert await page.locator('.pinned-anchor').count()==0
        await page.locator('.word-pin').nth(0).click();await page.locator('.word-pin').nth(4).click();await page.wait_for_timeout(600)
        if args.report:
            await page.screenshot(path=str(Path(args.report).with_suffix('.desktop.png')),full_page=True)
            await page.locator('.branch-workspace').screenshot(path=str(Path(args.report).with_suffix('.graph.png')))
        result['search_ms']=await page.evaluate('performance.getEntriesByName("wikilangs:pin-search").map(x=>x.duration)')
        assert not errors,errors
        # Blocking script.js is expected on a production build; no event can be sent.
        assert not [u for u in tracking if '/api/send' in u]
        result.update(errors=errors,analytics_blocked=len(tracking),browser=browser.version)
        if args.report:Path(args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False,indent=2));await browser.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base-url',default='http://127.0.0.1:4326');parser.add_argument('--report');parser.add_argument('--release');parser.add_argument('--expect-analytics',action='store_true');asyncio.run(run(parser.parse_args()))
