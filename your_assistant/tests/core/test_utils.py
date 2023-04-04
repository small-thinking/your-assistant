"""Test the utils.
Run this test with command: pytest your_assistant/tests/core/test_utils.py
"""
import os
import textwrap

import pytest

import your_assistant.core.utils as utils


@pytest.mark.parametrize(
    "input, chunk_size, expected",
    [
        (
            list("Hello world"),
            3,
            [("H", "e", "l"), ("l", "o", " "), ("w", "o", "r"), ("l", "d")],
        ),
        (range(14), 3, [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13)]),
    ],
)
def test_chunk_list(input, chunk_size, expected):
    chunk_iterator = utils.chunk_list(input, chunk_size)
    assert list(chunk_iterator) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            textwrap.dedent(
                """
                <?xml version='1.0' encoding='utf-8'?>
                <!DOCTYPE html>
                <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" epub:prefix="z3998: http://www.daisy.org/z3998/2012/vocab/structure/#" lang="en" xml:lang="en">
                <head/>
                    <body>
                        <div class="Basic-Text-Frame">
                            <p class="BodyTab">This is a test of "text".</p>
                            <p id="toc_marker-7" class="title-spacebefore">**CHAPTER 1: The first chapter**</p>
                            <p id="toc_marker-7" class="title-spacebefore">Normal paragraph.</p>
                            <p class="toc"><a href="part0053.html#FBATR_ebook">Link in a paragraph.</a></p>
                            <hr class="calibre1"/>
                            <ul class="calibre2">
                                <li class="Bullets"><a href="part0005.html#FBATR_ebook">Point 1.</a></li>
                                <li class="Bullets"><a href="part0006.html#FBATR_ebook">Point *2*</a></li>
                                <li class="Bullets"><a href="part0007.html#FBATR_ebook">Point "3"</a></li>
                                <li class="Bullets"><a href="part0008.html#FBATR_ebook">Point '4'.</a></li>
                                <li class="Bullets"><a href="part0009.html#FBATR_ebook">Last **point**.</a></li>
                            </ul>
                        </div>
                    </body>
                </html>
            """
            ),
            textwrap.dedent(
                """
                This is a test of "text".
                **CHAPTER 1: The first chapter**
                Normal paragraph.
                [Link in a paragraph.](part0053.html#FBATR_ebook)
                ---


                - [Point 1.](part0005.html#FBATR_ebook)
                - [Point *2*](part0006.html#FBATR_ebook)
                - [Point "3"](part0007.html#FBATR_ebook)
                - [Point '4'.](part0008.html#FBATR_ebook)
                - [Last **point**.](part0009.html#FBATR_ebook)
                """
            ),
        ),
    ],
)
def test_xml_to_markdown(input, expected):
    assert utils.xml_to_markdown(input).strip() == expected.strip()
