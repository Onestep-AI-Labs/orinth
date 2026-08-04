"use client";

import { forwardRef } from "react";
import Link from "next/link";
import { ChevronDown } from "lucide-react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "./cn";

export const buttonVariants = cva("", {
  variants: {
    variant: {
      primary: "primary-button",
      secondary: "secondary-button",
      ghost: "ghost-button",
      danger: "danger-button"
    },
    size: {
      sm: "button-sm",
      md: ""
    }
  },
  defaultVariants: {
    variant: "primary",
    size: "md"
  }
});

type ButtonVariantProps = VariantProps<typeof buttonVariants>;

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & ButtonVariantProps;

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant, size, className, type = "button", ...props },
  ref
) {
  return <button ref={ref} type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />;
});

export type ButtonLinkProps = React.ComponentProps<typeof Link> & ButtonVariantProps;

export function ButtonLink({ variant, size, className, ...props }: ButtonLinkProps) {
  return <Link className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}

export type IconButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  "aria-label": string;
  danger?: boolean;
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { danger, className, type = "button", ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(danger ? "danger-icon-button icon-button" : "icon-button", className)}
      title={props.title ?? props["aria-label"]}
      {...props}
    />
  );
});

export type SelectProps = React.SelectHTMLAttributes<HTMLSelectElement>;

/**
 * The single dropdown control.
 *
 * Every `<select>` in the app renders through this, so a dropdown looks the
 * same everywhere and matches the `TaskSelect` trigger: same 40px height,
 * `--radius-sm`, `--line-input` border, padding, and a single chevron on the
 * right. Previously each `<select>` inherited whichever of `.field select`,
 * `.select-label select`, `.bulk-bar select`, `.project-switcher select`,
 * `.dataset-pagination-size select`, or `.annotation-actions select` its
 * container supplied — three different border tokens between them — and every
 * one of them still rendered the platform's default double-arrow stepper,
 * which is what made them read as a different control from the Task picker.
 *
 * The native `<select>` is kept (its menu, keyboard behaviour, and screen
 * reader semantics are free and correct); `appearance: none` plus an overlaid
 * chevron is what aligns it with the custom trigger.
 *
 * Per DESIGN.md §2 the border is `--line-input`: the one weight that clears
 * WCAG 1.4.11 for a control boundary.
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { className, ...props },
  ref
) {
  return (
    <span className="select-shell">
      <select ref={ref} className={cn("select-control", className)} {...props} />
      {/* Presentational: the native control still owns all interaction, so the
          chevron must never swallow a click. */}
      <ChevronDown className="select-chevron" size={16} aria-hidden="true" />
    </span>
  );
});

export const badgeVariants = cva("badge", {
  variants: {
    tone: {
      neutral: "",
      ok: "badge-ok",
      fail: "badge-fail",
      warn: "badge-warn",
      info: "badge-info"
    }
  },
  defaultVariants: {
    tone: "neutral"
  }
});

export type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>;

export function Badge({ tone, className, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
