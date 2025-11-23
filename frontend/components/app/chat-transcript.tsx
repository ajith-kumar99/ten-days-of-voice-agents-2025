'use client';

import { AnimatePresence, type HTMLMotionProps, motion } from 'motion/react';
import { type ReceivedChatMessage } from '@livekit/components-react';
import { ChatEntry } from '@/components/livekit/chat-entry';

const MotionContainer = motion.create('div');
const MotionChatEntry = motion.create(ChatEntry);

const CONTAINER_MOTION_PROPS = {
  variants: {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  },
  initial: false,     // <-- IMPORTANT FIX
  animate: 'visible',
  exit: 'hidden',
};

const MESSAGE_MOTION_PROPS = {
  variants: {
    hidden: {
      opacity: 0,
      translateY: 10,
    },
    visible: {
      opacity: 1,
      translateY: 0,
    },
  },
};

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: ReceivedChatMessage[];
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  ...props
}: ChatTranscriptProps & Omit<HTMLMotionProps<'div'>, 'ref'>) {
  return (
    <AnimatePresence mode="popLayout">
      {!hidden && (
        <MotionContainer {...CONTAINER_MOTION_PROPS} {...props}>
          {messages.map(({ id, timestamp, from, message, editTimestamp }) => {
            const locale = navigator?.language ?? 'en-US';
            const messageOrigin = from?.isLocal ? 'local' : 'remote';
            const hasBeenEdited = !!editTimestamp;

            return (
              <MotionChatEntry
                key={id}
                locale={locale}
                timestamp={timestamp}
                message={message}
                messageOrigin={messageOrigin}
                hasBeenEdited={hasBeenEdited}
                variants={MESSAGE_MOTION_PROPS.variants}
                initial="hidden"     // <-- FIX
                animate="visible"    // <-- FIX
                exit="hidden"        // <-- FIX
              />
            );
          })}
        </MotionContainer>
      )}
    </AnimatePresence>
  );
}
