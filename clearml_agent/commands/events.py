from __future__ import print_function

import json
import time

from future.builtins import super

from trains_agent.commands.base import ServiceCommandSection
from trains_agent.helper.base import return_list


class Events(ServiceCommandSection):
    max_packet_size = 1024 * 1024
    max_event_size = 64 * 1024

    def __init__(self, *args, **kwargs):
        super(Events, self).__init__(*args, **kwargs)

    @property
    def service(self):
        """ Events command service endpoint """
        return 'events'

    def send_events(self, list_events):
        def send_packet(jsonlines):
            if not jsonlines:
                return 0
            num_lines = len(jsonlines)
            jsonlines = '\n'.join(jsonlines)

            new_events = self.post('add_batch', data=jsonlines, headers={'Content-type': 'application/json-lines'})
            if new_events['added'] != num_lines:
                print('Error (%s) sending events only %d of %d registered' %
                      (new_events['errors'], new_events['added'], num_lines))
                return int(new_events['added'])
            # print('Sent %d events' % num_lines)
            return num_lines

        # json every line and push into list of json strings
        count_bytes = 0
        lines = []
        sent_events = 0
        for i, event in enumerate(list_events):
            line = json.dumps(event)
            line_len = len(line) + 1
            if count_bytes + line_len > self.max_packet_size:
                # flush packet, and restart
                sent_events += send_packet(lines)
                count_bytes = 0
                lines = []

            count_bytes += line_len
            lines.append(line)

        # flush leftovers
        sent_events += send_packet(lines)
        # print('Sending events done: %d / %d events sent' % (sent_events, len(list_events)))
        return sent_events

    def send_log_events(self, worker_id, task_id, lines, level='DEBUG'):
        log_events = []
        base_timestamp = int(time.time() * 1000)
        base_log_items = {
            'type': 'log',
            'level': level,
            'task': task_id,
            'worker': worker_id,
        }

        def get_event(c):
            d = base_log_items.copy()
            d.update(msg=msg, timestamp=base_timestamp + c)
            return d

        # break log lines into event packets
        msg = ''
        count = 0
        for l in return_list(lines):
            # HACK ignore terminal reset ANSI code
            if l == '\x1b[0m':
                continue
            while l:
                if len(msg) + len(l) < self.max_event_size:
                    msg += l
                    l = None
                else:
                    left_over = self.max_event_size - len(msg)
                    msg += l[:left_over]
                    l = l[left_over:]
                    log_events.append(get_event(count))
                    msg = ''
                count += 1
        if msg:
            log_events.append(get_event(count))

        # now send the events
        return self.send_events(list_events=log_events)
