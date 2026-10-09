import emblem from '../assets/nat-shin-naung-emblem.png'

export default function BrandLogo({ className = '' }) {
  return (
    <img
      className={`brand-emblem ${className}`.trim()}
      src={emblem}
      alt=""
      aria-hidden="true"
    />
  )
}
