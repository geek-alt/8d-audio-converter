declare module 'react' {
    export const useState: any;
    export const useRef: any;
    export const useEffect: any;
    const React: any;
    export default React;
}

declare module 'lucide-react' {
    export const Upload: any;
    export const Music: any;
    export const Download: any;
    export const Play: any;
    export const Pause: any;
    export const Settings: any;
    export const Sliders: any;
    export const Zap: any;
    export const Waves: any;
    export const Server: any;
    export const Chrome: any;
    export const AlertCircle: any;
}

declare namespace JSX {
    interface IntrinsicElements {
        [elemName: string]: any;
    }
}
