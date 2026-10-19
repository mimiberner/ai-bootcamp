"""Embed results.json into report_template.html -> report.html."""
import json

data = json.load(open("results.json"))
html = open("report_template.html").read().replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
open("report.html", "w").write(html)
print("Wrote report.html")
