import logging
import cv2
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, List

import easyocr


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ScreenshotDriver:
    """A thin wrapper around a Selenium WebDriver that saves a screenshot before
    each important action. Screenshots are stored in a `screenshots/` folder
    with a sequential prefix and timestamp for easy inspection.
    """

    def __init__(self, driver: webdriver.Firefox, out_dir: Path = Path("screenshots")):
        self.driver = driver
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self.history: List[Path] = []

    def _next_name(self, label: Optional[str] = None) -> Path:
        self._counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if label:
            # Sanitize label for filesystem
            label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
        else:
            label = "action"
        name = f"{self._counter:04d}_{ts}_{label}.png"
        return self.out_dir / name

    def screenshot(self, label: Optional[str] = None) -> Path:
        path = self._next_name(label)
        self.driver.save_screenshot(str(path))
        logging.info("Saved screenshot: %s", path)
        self.history.append(path)
        return path

    def get(self, url: str) -> None:
        self.screenshot(f"before_get_{url}")
        return self.driver.get(url)

    def find_element(self, *args, label: Optional[str] = None, **kwargs):
        self.screenshot(label or "before_find_element")
        el = self.driver.find_element(*args, **kwargs)
        return WrappedElement(el, self, label=label)

    def find_elements(
        self, *args, label: Optional[str] = None, **kwargs
    ) -> List["WrappedElement"]:
        self.screenshot(label or "before_find_elements")
        els = self.driver.find_elements(*args, **kwargs)
        return [WrappedElement(e, self, label=label) for e in els]

    def implicitly_wait(self, seconds: float) -> None:
        return self.driver.implicitly_wait(seconds)

    def quit(self) -> None:
        self.screenshot("before_quit")
        return self.driver.quit()

    def __getattr__(self, item: str) -> Any:
        return getattr(self.driver, item)


class WrappedElement:
    """Wrap a WebElement so we can take a screenshot before interactions like
    click() and send_keys()."""

    def __init__(
        self,
        element: WebElement,
        sdriver: ScreenshotDriver,
        label: Optional[str] = None,
    ):
        self._element = element
        self._sdriver = sdriver
        self._label = label

    def click(self):
        self._sdriver.screenshot(self._label or "before_click")
        return self._element.click()

    def send_keys(self, *keys):
        self._sdriver.screenshot(
            self._label or f"before_send_keys_{'_'.join(map(str, keys))}"
        )
        return self._element.send_keys(*keys)

    def submit(self):
        self._sdriver.screenshot(self._label or "before_submit")
        return self._element.submit()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._element, item)


def main() -> None:
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    raw_driver = webdriver.Firefox(options=options)
    driver = ScreenshotDriver(raw_driver)
    reader = easyocr.Reader(["en"])

    try:
        driver.get("https://en.wikipedia.org")
        search_box = driver.find_element(By.ID, "searchInput", label="search_input")
        search_box.send_keys("Napoleon")
        search_box.submit()
    finally:
        driver.quit()

    for img_path in driver.history:
        logging.info("Annotating %s", img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            logging.error("Could not read image %s", img_path)
            continue

        results = reader.readtext(str(img_path))
        for bbox, text, prob in results:
            # bbox structure: [[x, y], [x, y], [x, y], [x, y]]
            tl = tuple(map(int, bbox[0]))
            br = tuple(map(int, bbox[2]))
            cv2.rectangle(img, tl, br, (0, 255, 0), 2)
            cv2.putText(
                img,
                text,
                (tl[0], tl[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        annotated_path = img_path.with_name(f"annotated_{img_path.name}")
        cv2.imwrite(str(annotated_path), img)
        logging.info("Saved annotated screenshot: %s", annotated_path)


if __name__ == "__main__":
    main()
