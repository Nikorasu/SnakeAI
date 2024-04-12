#!/usr/bin/env python3
from time import sleep
from threading import Thread

# Just a simple terminal loading animation -Nik

class LoadingAnim:
    def __init__(self,full=True):
        self.anim = [
            "⢀⠀",
            "⡀⠀",
            "⠄⠀",
            "⢂⠀",
            "⡂⠀",
            "⠅⠀",
            "⢃⠀",
            "⡃⠀",
            "⠍⠀",
            "⢋⠀",
            "⡋⠀",
            "⠍⠁",
            "⢋⠁",
            "⡋⠁",
            "⠍⠉",
            "⠋⠉",
            "⠋⠉",
            "⠉⠙",
            "⠉⠙",
            "⠉⠩",
            "⠈⢙",
            "⠈⡙",
            "⢈⠩",  #loopstart 22
            "⡀⢙",
            "⠄⡙",
            "⢂⠩",
            "⡂⢘",
            "⠅⡘",
            "⢃⠨",
            "⡃⢐",
            "⠍⡐",
            "⢋⠠",
            "⡋⢀",
            "⠍⡁",
            "⢋⠁",
            "⡋⠁",
            "⠍⠉",
            "⠋⠉",
            "⠋⠉",
            "⠉⠙",
            "⠉⠙",
            "⠉⠩",
            "⠈⢙",
            "⠈⡙",  #loopend 43
            "⠈⠩",
            "⠀⢙",
            "⠀⡙",
            "⠀⠩",
            "⠀⢘",
            "⠀⡘",
            "⠀⠨",
            "⠀⢐",
            "⠀⡐",
            "⠀⠠",
            "⠀⢀",
            "⠀⡀",
            "  "
        ] if full else [
            "⢄",
            "⢂",
            "⢁",
            "⡁",
            "⡈",
            "⡐",
            "⡠",
            " "
        ]
        self.loopstart = 22 if full else 0
        self.loopend = 43 if full else 6

    def start(self):
        self.thread = Thread(target=self._run)
        self.thread.daemon = True  # Set the thread to daemon
        self.thread.start()

    def _run(self):
        indx = 0
        self.loop = True
        print('\033[?25l', end='')
        while indx < len(self.anim):
            print(self.anim[indx], end='\x1b[D' if self.loopend == 6 else '\x1b[2D', flush=True)
            sleep(.1)
            if self.loop and indx == self.loopend:
                indx = self.loopstart
            indx += 1
        print('\033[?25h', end='')

    def stop(self):
        self.loop = False
        self.thread.join() # Wait for the thread to finish


if __name__ == '__main__':
    print('Loading... ',end='')
    la = LoadingAnim()
    la.start()
    sleep(10)
    la.stop()
    print('Done!')