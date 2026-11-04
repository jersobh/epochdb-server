import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { TypeaheadItem } from '../types';
import { typeColor } from '../lib/graph';

interface TypeaheadInputProps {
  id?: string;
  className?: string;
  placeholder?: string;
  value: string;
  onChange: (value: string) => void;
  getItems: (query: string) => TypeaheadItem[];
  onPick?: (item: TypeaheadItem) => void;
  onSubmit?: (value: string) => void;
  type?: string;
  autoComplete?: string;
}

export function TypeaheadInput({
  id,
  className = 'ds-input',
  placeholder,
  value,
  onChange,
  getItems,
  onPick,
  onSubmit,
  type = 'text',
  autoComplete = 'off',
}: TypeaheadInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<TypeaheadItem[]>([]);
  const [idx, setIdx] = useState(-1);
  const [coords, setCoords] = useState<{
    left: number;
    top?: number;
    bottom?: number;
    width: number;
    maxHeight: number;
  } | null>(null);

  const hide = () => {
    setOpen(false);
    setIdx(-1);
  };

  const updatePosition = () => {
    if (!inputRef.current) return;
    const r = inputRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - r.bottom - 8;
    const spaceAbove = r.top - 8;
    const useAbove = spaceBelow < 180 && spaceAbove > spaceBelow;

    setCoords({
      left: r.left,
      width: r.width,
      ...(useAbove
        ? {
            bottom: window.innerHeight - r.top + 4,
            maxHeight: Math.min(260, spaceAbove),
          }
        : {
            top: r.bottom + 4,
            maxHeight: Math.min(260, Math.max(100, spaceBelow)),
          }),
    });
  };

  const render = (q: string) => {
    const next = (getItems(q) || []).slice(0, 8);
    setItems(next);
    setIdx(-1);
    if (!next.length) {
      setOpen(false);
      return;
    }
    updatePosition();
    setOpen(true);
  };

  const pick = (it: TypeaheadItem) => {
    onChange(it.value);
    hide();
    onPick?.(it);
  };

  useLayoutEffect(() => {
    if (open) {
      updatePosition();
    }
  }, [open, items]);

  useEffect(() => {
    const onScroll = (e: Event) => {
      if (listRef.current && listRef.current.contains(e.target as Node)) {
        return;
      }
      hide();
    };
    window.addEventListener('resize', hide);
    document.addEventListener('scroll', onScroll, true);
    return () => {
      window.removeEventListener('resize', hide);
      document.removeEventListener('scroll', onScroll, true);
    };
  }, []);

  return (
    <>
      <input
        ref={inputRef}
        id={id}
        className={className}
        type={type}
        placeholder={placeholder}
        value={value}
        autoComplete={autoComplete}
        spellCheck={false}
        onChange={(e) => {
          onChange(e.target.value);
          render(e.target.value);
        }}
        onFocus={() => render(value)}
        onBlur={() => setTimeout(hide, 120)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown' && open) {
            e.preventDefault();
            setIdx((i) => (i + 1) % items.length);
          } else if (e.key === 'ArrowUp' && open) {
            e.preventDefault();
            setIdx((i) => (i - 1 + items.length) % items.length);
          } else if (e.key === 'Escape') {
            hide();
          } else if (e.key === 'Tab' && open && idx >= 0) {
            e.preventDefault();
            pick(items[idx]);
          } else if (e.key === 'Enter') {
            if (open && idx >= 0) {
              e.preventDefault();
              pick(items[idx]);
            } else {
              hide();
              onSubmit?.(value);
            }
          }
        }}
      />
      {open && items.length > 0 && coords
        ? createPortal(
            <div
              ref={listRef}
              className="ta-list open"
              style={{
                left: `${coords.left}px`,
                width: `${coords.width}px`,
                ...(coords.top !== undefined
                  ? { top: `${coords.top}px`, bottom: 'auto' }
                  : { bottom: `${coords.bottom}px`, top: 'auto' }),
                maxHeight: `${coords.maxHeight}px`,
              }}
            >
              {items.map((it, i) => (
                <button
                  key={`${it.value}-${i}`}
                  type="button"
                  className={`ta-item${i === idx ? ' active' : ''}`}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    pick(it);
                  }}
                >
                  {it.type ? (
                    <span className="ta-dot" style={{ background: typeColor(it.type) }} />
                  ) : null}
                  <span className="ta-label">{it.label}</span>
                  <span className="ta-tag">{it.tag || ''}</span>
                </button>
              ))}
            </div>,
            document.body
          )
        : null}
    </>
  );
}
