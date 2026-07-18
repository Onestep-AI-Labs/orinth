"use client";

import { forwardRef } from "react";
import Link from "next/link";
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
