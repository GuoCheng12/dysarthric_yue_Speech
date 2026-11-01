#!/usr/bin/env python3
"""Apply the frozen literal-delivery reporter to the new contextual execution."""
import mas_contextual_matched_sg_probe_v2 as contextual
import report_mas_literal_matched_sg_v1 as report

assert report.run.ROOT == contextual.ROOT

if __name__ == '__main__':
    report.main()
